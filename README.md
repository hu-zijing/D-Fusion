# D-Fusion

python3 run_sample.py --config.exp_name test --config.sample.num_batches_per_epoch 2 --config.sample.batch_size 2 --config.dev_id 0 --config.pretrained.model ../../model/model/stablediffusion/sdv2.1-base --config.train.num_epochs 1 --config.prompt_file config/prompt/template1_3prompt.json --config.mask_thr_file config/prompt/template1_mask.json --config.item_idx_file config/prompt/template1_item.json > log/log_test

python3 run_eval.py --config.dev_id 0 --config.exp_name test
