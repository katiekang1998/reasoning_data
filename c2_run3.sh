CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-440 --eval_type test
CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs10/checkpoint-440 --eval_type train_aug_subsample



# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-7488 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-7488 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-8320 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs20/checkpoint-8320 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_all_2epochs_llama2 --eval_type train_aug_2 --seed 3
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_all_2epochs_llama2 --eval_type test --seed 3