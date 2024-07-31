
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-220 --eval_type test
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-220 --eval_type train_aug_subsample



# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_total10000_epochs5/checkpoint-416 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_total10000_epochs5/checkpoint-416 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_total10000_epochs5/checkpoint-833 --eval_type train_aug_subsample



# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-832 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-832 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-1664 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-1664 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-2496 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-2496 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-281 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-281 --eval_type train

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-562 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-562 --eval_type train


# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_all_2epochs_llama2 --eval_type train_aug_1 --seed 2
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_all_2epochs_llama2 --eval_type test --seed 0