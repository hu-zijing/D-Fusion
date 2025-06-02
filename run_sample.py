import contextlib
import os
import datetime
import time
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
import torch
from torchvision.utils import save_image
from functools import partial
import tqdm
from PIL import Image
import json
import pickle
import random
import importlib
import tree
from utils.utils import seed_everything

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/config.py", "Sampling configuration.")

logger = get_logger(__name__)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config
    debug_idx = 0
    print(f'========== seed: {config.seed} ==========')
    torch.cuda.set_device(config.dev_id)

    unique_id = config.exp_name if config.exp_name else datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    save_dir = os.path.join(config.save_path, unique_id)

    seed_everything(config.seed)

    prompt_list = [config.prompt]
    if len(config.prompt_file)!=0:
        with open(config.prompt_file, 'r') as f:
            prompt_list = json.load(f)
    print('prompt list:', prompt_list)
    prompt_cnt = len(prompt_list)
    total_num_batches_per_epoch = config.sample.num_batches_per_epoch*prompt_cnt

    mask_thr_list = [config.mask_thr for _ in range(prompt_cnt)]
    if len(config.mask_thr_file)!=0:
        with open(config.mask_thr_file, 'r') as f:
            mask_thr_list = json.load(f)
            mask_thr_list = [mask_thr_list[p] for p in prompt_list]
    item_idx_list = [config.item_idx for _ in range(prompt_cnt)]
    if len(config.item_idx_file)!=0:
        with open(config.item_idx_file, 'r') as f:
            item_idx_list = json.load(f)
            item_idx_list = [item_idx_list[p] for p in prompt_list]

    if config.resume_from:
        print("loading model. Please Wait.")
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        print("load successfully!")

    accelerator_config = ProjectConfiguration(
        project_dir=save_dir, 
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config, 
        gradient_accumulation_steps = config.train.gradient_accumulation_steps*config.sample.num_steps*config.sample.batch_size*total_num_batches_per_epoch//config.train.batch_size//2
    )

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.float16) # float16
    # pipeline = StableDiffusionPipeline.from_single_file(config.pretrained.model, torch_dtype=torch.float16)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    # total_image_num_per_gpu = config.sample.batch_size * config.sample.num_batches_per_epoch
    global_idx = 0 # accelerator.process_index * total_image_num_per_gpu 
    local_idx = 0
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        pipeline.unet.set_attn_processor(lora_attn_procs)
        trainable_layers = AttnProcsLayers(pipeline.unet.attn_processors)
    else:
        trainable_layers = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        # print(models)
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            # print(models[0].state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model
    
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers = accelerator.prepare(trainable_layers)

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)

    # torch.cuda.empty_cache()
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    inject_component = list(map(lambda x: True if x=='T' else False, config.inject_component.split(',')))
    inject_t = [(tuple(map(int, x.split('-'))) if x else None) for x in config.inject_t.split(',')]
    inject_l = [(tuple(map(int, x.split('-'))) if x else None) for x in config.inject_l.split(',')]

    for epoch in range(config.train.begin_epoch, config.train.num_epochs):
        pipeline.unet.eval()
        prompts = []
        samples = []

        print(f'========== Epoch {epoch} ==========')
        save_dir_epoch = os.path.join(save_dir, f'epoch{epoch}')

        print(f'======== Sampling ========')

        for idx in tqdm(
            range(total_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # we set the prompts to be the same
            prompts1 = [
                prompt_list[idx//config.sample.num_batches_per_epoch]
                for _ in range(config.sample.batch_size)
                ] 
            prompts.extend(prompts1)
            # encode prompts
            prompt_ids1 = pipeline.tokenizer(
                prompts1,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)

            prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]

            # combine prompt and neg_prompt
            prompt_embeds1_combine = pipeline._encode_prompt(
                None,
                accelerator.device,
                1,
                config.sample.cfg,
                None,
                prompt_embeds=prompt_embeds1,
                negative_prompt_embeds=sample_neg_prompt_embeds
            )

            gs = [torch.Generator(device='cuda') for _ in range(config.sample.batch_size)]
            for i,g in enumerate(gs): 
                g.manual_seed(config.seed+epoch*total_num_batches_per_epoch*config.sample.batch_size+idx*config.sample.batch_size+i)
            noise_latents1 = pipeline.prepare_latents(
                config.sample.batch_size, 
                pipeline.unet.config.in_channels, ## channels
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## height
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## width
                prompt_embeds1.dtype, 
                accelerator.device, 
                gs ## generator
            )
            pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
            ts = pipeline.scheduler.timesteps
            extra_step_kwargs = pipeline.prepare_extra_step_kwargs(gs, config.sample.eta)

            latents_t = noise_latents1
            latents = [noise_latents1]
            log_probs = []
            
            for i, t in tqdm(
                enumerate(ts),
                desc="Timestep",
                position=3,
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):  
                # sample
                with autocast():
                    with torch.no_grad():
                        latents_input = torch.cat([latents_t] * 2) if config.sample.cfg else latents_t
                        latents_input = pipeline.scheduler.scale_model_input(latents_input, t)

                        noise_pred, attn_probs, _ = pipeline.unet(
                            latents_input,
                            t,
                            encoder_hidden_states=prompt_embeds1_combine,
                            return_dict=False,
                        )
                        noise_pred = noise_pred[0]

                        if config.sample.cfg:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        latents_t_1, log_prob, _ = ddim_step_with_logprob(pipeline.scheduler, noise_pred, t, latents_t, **extra_step_kwargs)

                        latents_t = latents_t_1
                        latents.append(latents_t)
                        log_probs.append(log_prob)

            images = latents_decode(pipeline, latents_t, accelerator.device, prompt_embeds1.dtype).cpu().detach()
            os.makedirs(os.path.join(save_dir_epoch, "images"), exist_ok=True)
            global_idx = len(os.listdir(os.path.join(save_dir_epoch, "images")))
            for j, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil.save(os.path.join(save_dir_epoch, f"images/{(j+global_idx):05}.png"))
            with open(os.path.join(save_dir_epoch, f'prompt.json'),'w') as f:
                json.dump(prompts, f)

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            prompt_embeds = prompt_embeds1
            current_latents = latents[:, :-1]
            next_latents = latents[:, 1:]
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)
            log_probs = torch.stack(log_probs, dim=1)
            samples.append(
                {
                    "prompt_embeds": prompt_embeds.cpu().detach(),
                    "timesteps": timesteps.cpu().detach(),
                    "latents": current_latents.cpu().detach(),  # each entry is the latent before timestep t
                    "next_latents": next_latents.cpu().detach(),  # each entry is the latent after timestep t
                    "log_probs": log_probs.cpu().detach()
                }
            )

        # wait to be evaluated
        cnt_file = os.path.join(os.path.join(save_dir_epoch, 'eval', f'clip_scores.txt'))
        evaled_file = os.path.join(os.path.join(save_dir_epoch, 'eval', f'clip_scores.pkl'))
        while True: 
            evaled_cnt = 0
            if os.path.exists(cnt_file):
                try: 
                    with open(cnt_file, 'r') as f:
                        evaled_cnt = json.load(f)
                except Exception as e: 
                    # if the file is being written, exception will occur
                    continue
            if evaled_cnt >= total_num_batches_per_epoch * config.sample.batch_size: 
                break
            time.sleep(2)
        with open(evaled_file, 'rb') as f:
            eval_scores = pickle.load(f)

        init_samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        init_samples['eval_scores'] = eval_scores

        print(f'======== Sample Injecting ========')
        prompt_map = {k:[] for k in prompt_list}
        for i, p in enumerate(prompts): 
            prompt_map[p].append(i)

        total_batch_size = init_samples["eval_scores"].shape[0]
        top_used_factor = 2  ## only use 50%/top_used_factor best sample to inject
        score_order1 = []
        score_order2 = []
        for k,v in prompt_map.items(): 
            if len(v) != 0: 
                score_order_p = eval_scores[v].argsort(descending=True)
                score_order_p = torch.tensor(v)[score_order_p]
                score_order1.append(score_order_p[:len(v)//(2*top_used_factor)].repeat_interleave(top_used_factor,dim=0))
                score_order2.append(score_order_p[len(v)//2:])
        score_order = torch.cat(score_order1+score_order2)
        samples = {k: v[score_order] for k, v in init_samples.items()}

        order_list = score_order.tolist()
        print(order_list)
        inject_prompts = [prompts[i] for i in order_list[total_batch_size//2:]]
        with open(os.path.join(save_dir_epoch, 'injected_prompt.json'), 'w') as f: 
            json.dump(inject_prompts, f)
        with open(os.path.join(save_dir_epoch, 'eval', 'injected_map.json'), 'w') as f: 
            json.dump({i:(o,order_list[i]) for i,o in enumerate(order_list[total_batch_size//2:])}, f)
        
        injected_samples = []
        for idx in tqdm(
            range(total_num_batches_per_epoch//2),
            desc=f"Epoch {epoch}: inject sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # combine prompt and neg_prompt
            prompt_embeds1 = samples['prompt_embeds'][total_batch_size//2+idx*config.sample.batch_size : total_batch_size//2+(idx+1)*config.sample.batch_size].to(accelerator.device)
            prompt_embeds1_combine = pipeline._encode_prompt(
                None,
                accelerator.device,
                1,
                config.sample.cfg,
                None,
                prompt_embeds=prompt_embeds1,
                negative_prompt_embeds=sample_neg_prompt_embeds
            )
            gs1 = [torch.Generator(device='cuda') for _ in range(config.sample.batch_size)]
            for i,g in enumerate(gs1): 
                g.manual_seed(config.seed+epoch*total_batch_size+score_order[total_batch_size//2+idx*config.sample.batch_size+i].item())
            noise_latents1 = pipeline.prepare_latents(
                config.sample.batch_size, 
                pipeline.unet.config.in_channels, ## channels
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## height
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## width
                prompt_embeds1.dtype, 
                accelerator.device, 
                gs1 ## generator
            )
            extra_step_kwargs1 = pipeline.prepare_extra_step_kwargs(gs1, config.sample.eta)

            # combine prompt and neg_prompt
            prompt_embeds3 = samples['prompt_embeds'][total_batch_size//2+idx*config.sample.batch_size : total_batch_size//2+(idx+1)*config.sample.batch_size].to(accelerator.device)
            prompt_embeds3_combine = pipeline._encode_prompt(
                None,
                accelerator.device,
                1,
                config.sample.cfg,
                None,
                prompt_embeds=prompt_embeds3,
                negative_prompt_embeds=sample_neg_prompt_embeds
            )
            gs3 = [torch.Generator(device='cuda') for _ in range(config.sample.batch_size)]
            for i,g in enumerate(gs3): 
                g.manual_seed(config.seed+epoch*total_batch_size+score_order[total_batch_size//2+idx*config.sample.batch_size+i].item())
            noise_latents3 = pipeline.prepare_latents(
                config.sample.batch_size, 
                pipeline.unet.config.in_channels, ## channels
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## height
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## width
                prompt_embeds3.dtype, 
                accelerator.device, 
                gs3 ## generator
            )
            extra_step_kwargs3 = pipeline.prepare_extra_step_kwargs(gs3, config.sample.eta)

            # combine prompt and neg_prompt
            prompt_embeds2 = samples['prompt_embeds'][idx*config.sample.batch_size : (idx+1)*config.sample.batch_size].to(accelerator.device)
            prompt_embeds2_combine = pipeline._encode_prompt(
                None,
                accelerator.device,
                1,
                config.sample.cfg,
                None,
                prompt_embeds=prompt_embeds2, 
                negative_prompt_embeds=sample_neg_prompt_embeds
            )
            gs2 = [torch.Generator(device='cuda') for _ in range(config.sample.batch_size)]
            for i,g in enumerate(gs2): 
                g.manual_seed(config.seed+epoch*total_batch_size+score_order[idx*config.sample.batch_size+i].item())
            noise_latents2 = pipeline.prepare_latents(
                config.sample.batch_size, 
                pipeline.unet.config.in_channels, ## channels
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## height
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## width
                prompt_embeds2.dtype, 
                accelerator.device, 
                gs2 ## generator
            )
            extra_step_kwargs2 = pipeline.prepare_extra_step_kwargs(gs2, config.sample.eta)

            pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
            ts = pipeline.scheduler.timesteps

            latents_t1 = noise_latents1
            latents_t3 = noise_latents3
            latents_t2 = noise_latents2
            latents = [noise_latents1]
            log_probs = []
            
            for i, t in tqdm(
                enumerate(ts),
                desc="Timestep",
                position=3,
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):  
                # sample
                # print(t.shape, ts)
                cur_inject_component = [ic and it[0]<=config.sample.num_steps-i-1<it[1] for ic,it in zip(inject_component,inject_t)]

                with autocast():
                    with torch.no_grad():
                        latents_input = torch.cat([latents_t2] * 2) if config.sample.cfg else latents_t2
                        latents_input = pipeline.scheduler.scale_model_input(latents_input, t)
                        noise_pred, attn_probs, cross_mask = pipeline.unet(
                            latents_input,
                            t,
                            encoder_hidden_states=prompt_embeds2_combine, # (prompt_embeds1_combine, prompt_embeds2_combine),
                            return_dict=False,
                            inject_component = cur_inject_component, 
                            inject_l = inject_l, 
                            item_idx = item_idx_list[score_order[idx*config.sample.batch_size]//(config.sample.num_batches_per_epoch*config.sample.batch_size)], 
                            mask_thr = mask_thr_list[score_order[idx*config.sample.batch_size]//(config.sample.num_batches_per_epoch*config.sample.batch_size)]
                        )
                        noise_pred = noise_pred[0]
                        if config.sample.cfg:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        latents_t_1, _, latents_0 = ddim_step_with_logprob(pipeline.scheduler, noise_pred, t, latents_t2, **extra_step_kwargs2)
                        latents_t2 = latents_t_1

                        os.makedirs(os.path.join(save_dir_epoch, "mask/"), exist_ok=True)
                        bsize, width_height, item_cnt = cross_mask.shape
                        width = int(width_height**0.5)
                        temp_cross_mask = cross_mask.permute(0,2,1).reshape(bsize,item_cnt,width,width)
                        global_idx = len(os.listdir(os.path.join(save_dir_epoch, "injected_images"))) if os.path.exists(os.path.join(save_dir_epoch, "injected_images")) else 0
                        for b in range(bsize):
                            for item in range(item_cnt):
                                save_image(
                                    temp_cross_mask[b,item].unsqueeze(0), 
                                    os.path.join(save_dir_epoch, f"mask/{(b+global_idx):05}_item{item}_t{i}.png")
                                )

                        latents_input = torch.cat([latents_t3] * 2) if config.sample.cfg else latents_t3
                        latents_input = pipeline.scheduler.scale_model_input(latents_input, t)
                        noise_pred, _, cross_mask2 = pipeline.unet(
                            latents_input,
                            t,
                            encoder_hidden_states=prompt_embeds3_combine, # (prompt_embeds1_combine, prompt_embeds2_combine),
                            return_dict=False,
                            inject_component = cur_inject_component, 
                            inject_l = inject_l, 
                            item_idx = item_idx_list[score_order[idx*config.sample.batch_size]//(config.sample.num_batches_per_epoch*config.sample.batch_size)], 
                            mask_thr = mask_thr_list[score_order[idx*config.sample.batch_size]//(config.sample.num_batches_per_epoch*config.sample.batch_size)]
                        )
                        noise_pred = noise_pred[0]
                        if config.sample.cfg:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        latents_t_1, _, latents_0 = ddim_step_with_logprob(pipeline.scheduler, noise_pred, t, latents_t3, **extra_step_kwargs3)
                        latents_t3 = latents_t_1

                        latents_input = torch.cat([latents_t1] * 2) if config.sample.cfg else latents_t1
                        latents_input = pipeline.scheduler.scale_model_input(latents_input, t)
                        noise_pred, _, _ = pipeline.unet(
                            latents_input,
                            t,
                            encoder_hidden_states=prompt_embeds1_combine, # (prompt_embeds1_combine, prompt_embeds2_combine),
                            return_dict=False,
                            inject_component = cur_inject_component, 
                            inject_l = inject_l, 
                            given_attn_map = attn_probs[0] if isinstance(attn_probs, tuple) else attn_probs, 
                            given_cross_mask = (cross_mask,cross_mask2),
                        )
                        noise_pred = noise_pred[0]
                        if config.sample.cfg:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        latents_t_1, log_prob1, latents_0 = ddim_step_with_logprob(pipeline.scheduler, noise_pred, t, latents_t1, **extra_step_kwargs1)
                        latents_t1 = latents_t_1
                        
                        latents.append(latents_t1)
                        log_probs.append(log_prob1)

            images = latents_decode(pipeline, latents_t1, accelerator.device, prompt_embeds1.dtype).cpu().detach()
            os.makedirs(os.path.join(save_dir_epoch, "injected_images"), exist_ok=True)
            global_idx = len(os.listdir(os.path.join(save_dir_epoch, "injected_images")))
            for j, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil.save(os.path.join(save_dir_epoch, f"injected_images/{(j+global_idx):05}.png"))

            images = latents_decode(pipeline, latents_t2, accelerator.device, prompt_embeds2.dtype).cpu().detach()
            os.makedirs(os.path.join(save_dir_epoch, "ref_images"), exist_ok=True)
            global_idx = len(os.listdir(os.path.join(save_dir_epoch, "ref_images")))
            for j, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil.save(os.path.join(save_dir_epoch, f"ref_images/{(j+global_idx):05}.png"))
            images = latents_decode(pipeline, latents_t3, accelerator.device, prompt_embeds3.dtype).cpu().detach()
            os.makedirs(os.path.join(save_dir_epoch, "base_images"), exist_ok=True)
            global_idx = len(os.listdir(os.path.join(save_dir_epoch, "base_images")))
            for j, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil.save(os.path.join(save_dir_epoch, f"base_images/{(j+global_idx):05}.png"))

if __name__ == "__main__":
    app.run(main)
