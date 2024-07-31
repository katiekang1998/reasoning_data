CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-880 --eval_type test
CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-880 --eval_type train_aug_subsample

CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-1100 --eval_type test


# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_total10000_epochs5/checkpoint-1666 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_total10000_epochs5/checkpoint-1666 --eval_type train_aug_subsample




# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-5824 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-5824 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-6656 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-6656 --eval_type train_aug_subsample



# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-1406 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-1406 --eval_type train

# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-1686 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_memorized/checkpoint-1686 --eval_type train


# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_all_2epochs_llama2 --eval_type train_aug_1 --seed 3
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_all_2epochs_llama2 --eval_type test --seed 2