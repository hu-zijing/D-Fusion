import os
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from functools import partial
import tqdm
from PIL import Image
import json
import importlib
import tqdm
import time
import open_clip
import numpy as np
import torch
import pickle

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/config.py", "Evaluation configuration.")

def main(_):
    config = FLAGS.config

    torch.cuda.set_device(config.dev_id)

    #################### set up eval model ####################
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='../../model/model/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin')
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    model = model.to(device)

    eval_path = os.path.join(config.save_path, config.exp_name)
    prefix_list = ['', 'injected_']

    while(True): 
        if not os.path.exists(eval_path): 
            time.sleep(2)
            continue

        for stage_id in os.listdir(eval_path): 
            if not os.path.isdir(os.path.join(eval_path, stage_id)): 
                continue

            for prefix in prefix_list: 
                save_dir = os.path.join(eval_path, stage_id, 'eval')
                image_dir = os.path.join(eval_path, stage_id, f'{prefix}images')
                evaled_file = os.path.join(save_dir, f'{prefix}clip_scores.pkl')
                cnt_file = os.path.join(save_dir, f'{prefix}clip_scores.txt')
                history_file = os.path.join(eval_path, f'history_scores.pkl')
                norm_file = os.path.join(save_dir, f'{prefix}clip_norm_scores.pkl')

                if not os.path.exists(image_dir): 
                    continue

                os.makedirs(save_dir, exist_ok=True)

                evaled_cnt = 0
                if os.path.exists(cnt_file): 
                    with open(cnt_file, 'r') as f:
                        evaled_cnt = json.load(f)
                else: 
                    with open(cnt_file, 'w') as f:
                        json.dump(evaled_cnt, f)
                
                image_list = os.listdir(image_dir) if os.path.exists(image_dir) else []
                prompts = []
                if os.path.exists(os.path.join(eval_path, stage_id, f'{prefix}prompt.json')):
                    try: 
                        with open(os.path.join(eval_path, stage_id, f'{prefix}prompt.json'), 'r') as f: 
                            prompts = json.load(f)
                    except Exception as e: 
                        # if the file is being written, exception will occur
                        continue
                image_cnt = min(len(image_list), len(prompts))

                if image_cnt <= evaled_cnt: 
                    # no new sample
                    continue

                evaled_scores = None
                if os.path.exists(evaled_file): 
                    with open(evaled_file, 'rb') as f:
                        evaled_scores = pickle.load(f)

                new_eval_cnt = 0
                similarity = []
                maximum_onetime = 8
                try:
                    for i in range(evaled_cnt, image_cnt, maximum_onetime): 
                        cur_eval_cnt = maximum_onetime if i+maximum_onetime <= image_cnt else (image_cnt-i)
                        image_input = torch.tensor(np.stack([preprocess(Image.open(os.path.join(image_dir, f'{image_idx:05}.png'))).numpy() for image_idx in range(i, i+cur_eval_cnt)])).to(device)
                        text_inputs = tokenizer(prompts[i:i+cur_eval_cnt]).to(device)

                        with torch.no_grad():
                            image_features = model.encode_image(image_input)
                            text_features = model.encode_text(text_inputs)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        similarity.append( (image_features @ text_features.T)[torch.arange(cur_eval_cnt), torch.arange(cur_eval_cnt)] )

                        new_eval_cnt += cur_eval_cnt
                except Exception as e:
                    continue

                similarity = torch.cat(similarity)

                R = similarity.cpu().detach()
                evaled_scores = R if evaled_scores is None else torch.cat([evaled_scores, R])

                if len(prefix) == 0: 
                    history_scores = {}
                    if os.path.exists(history_file):
                        with open(history_file, 'rb') as f: 
                            history_scores = pickle.load(f)

                    cur_data = {}
                    for i,p in enumerate(prompts[evaled_cnt:evaled_cnt+new_eval_cnt]):
                        if p in cur_data:
                            cur_data[p].append(R[i])
                        else:
                            cur_data[p] = [R[i]]
                    for k,v in cur_data.items():
                        if k in history_scores:
                            history_scores[k] = torch.cat([history_scores[k], torch.stack(v)])
                        else:
                            history_scores[k] = torch.stack(v)

                    with open(history_file, 'wb') as f: 
                        pickle.dump(history_scores, f)

                with open(os.path.join(save_dir, f'{prefix}clip_score_mean.json'), 'w') as f:
                    json.dump(evaled_scores.mean().item(), f)
                with open(evaled_file, 'wb') as f:
                    pickle.dump(evaled_scores, f)
                with open(cnt_file, 'w') as f:
                    json.dump(evaled_cnt+new_eval_cnt, f)

                print(eval_path, stage_id, prefix, evaled_cnt, new_eval_cnt)

        time.sleep(2)


if __name__ == "__main__":
    app.run(main)
