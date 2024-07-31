CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-660 --eval_type test
CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-660 --eval_type train_aug_subsample

CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-1100 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_total10000_epochs5/checkpoint-1250 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_total10000_epochs5/checkpoint-1250 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_total10000_epochs5/checkpoint-833 --eval_type test



# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-3328 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-3328 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-4160 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-4160 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-4992 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-4992 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-844 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-844 --eval_type train

# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-1125 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-1125 --eval_type train


# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_all_2epochs_llama2 --eval_type train_aug_2 --seed 2
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_all_2epochs_llama2 --eval_type test --seed 1