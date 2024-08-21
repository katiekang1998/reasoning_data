
CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized_lr5e-07_bs128/checkpoint-288 --eval_type train --num_samples 5
CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized_lr5e-07_bs128/checkpoint-288 --eval_type test --num_samples 50





# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-07_epochs12/checkpoint-2488 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-07_epochs12/checkpoint-2488 --eval_type train


# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-07_epochs12/checkpoint-3110 --eval_type train

# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-440 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-440 --eval_type train_aug_subsample



# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-7488 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-7488 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-8320 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-8320 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_all_2epochs_llama2 --eval_type train_aug_2 --seed 3
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_all_2epochs_llama2 --eval_type test --seed 3