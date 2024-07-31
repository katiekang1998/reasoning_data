
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-550 --eval_type test
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-441 --eval_type test
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-331 --eval_type test
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-220 --eval_type test
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-110 --eval_type test


CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-550 --eval_type train_aug_subsample
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-441 --eval_type train_aug_subsample
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-331 --eval_type train_aug_subsample
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-220 --eval_type train_aug_subsample
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-110 --eval_type train_aug_subsample


# torchrun --nproc_per_node=4 --master_port=1234 math_aug3_train.py --train_type unmemorized_eq_1 --num_epochs 5 --num_train_points 2646
# torchrun --nproc_per_node=4 --master_port=1234 math_aug3_train.py --train_type unmemorized_eq_2 --num_epochs 5 --num_train_points 2646
# torchrun --nproc_per_node=4 --master_port=1234 math_aug3_train.py --train_type unmemorized_eq_3 --num_epochs 5 --num_train_points 2646
# torchrun --nproc_per_node=4 --master_port=1234 math_aug3_train.py --train_type unmemorized_eq_4 --num_epochs 5 --num_train_points 2646



# # torchrun --nproc_per_node=4 --master_port=1234 math_aug3_train.py --train_type rand_50_unmemorized_eq_50 --num_epochs 10 --num_train_points 5292

# # torchrun --nproc_per_node=4 --master_port=1234 math_aug3_train.py --train_type rand2646 --num_epochs 20 --num_train_points 2646

# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_50_unmemorized_eq_50_total5292_epochs10/checkpoint-2200 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_50_unmemorized_eq_50_total5292_epochs10/checkpoint-2200 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_50_unmemorized_eq_50_total5292_epochs10/checkpoint-1760 --eval_type test


# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-1540 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-1540 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-1760 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-1760 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-1980 --eval_type test


# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs20/checkpoint-2496 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs20/checkpoint-2496 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-281 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-281 --eval_type train

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-562 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-562 --eval_type train

# NAME=math_aug2_unmemorized_total15533_epochs2
# find /data/katie_kang/reasoning_data/ckpts/${NAME} -maxdepth 2 -type f -name "*.safetensors" -delete

# torchrun --nproc_per_node=4 --master_port=1234 math_aug2_subsample_train.py --train_type rand --num_epochs 5
# python math_eval.py --ckpt_dir ckpts/math_aug2_rand_total15533_epochs5 --eval_type test --num_samples 5 --seed 2
# python math_eval.py --ckpt_dir ckpts/math_aug2_rand_total15533_epochs5 --eval_type train_aug --num_samples 4 --seed 2





# torchrun --nproc_per_node=4 --master_port=1234 math_aug2_subsample_train.py --train_type unmemorized
# torchrun --nproc_per_node=4 --master_port=1234 math_aug2_subsample_train.py --train_type rand

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard5_30000 --eval_type test --seed 0
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard10_30000 --eval_type test --seed 0
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_rand_30000 --eval_type test --seed 0

# # CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard5_10000 --eval_type test
# # CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard10_10000 --eval_type test


# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_rand_20000 --eval_type test --seed 1
# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_rand_20000 --eval_type test --seed 3
# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_rand_20000 --eval_type test --seed 4

# # torchrun --nproc_per_node=4 --master_port=1234 gsm8k_aug_llama2.py --num_train_points 20000 --data_type hard10
# # torchrun --nproc_per_node=4 --master_port=1234 gsm8k_aug_llama2.py --num_train_points 30000 --data_type hard5

# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_all_2epochs_llama2 --eval_type train_aug_1 --seed 2
# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_all_2epochs_llama2 --eval_type test --seed 0


# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_rand_5000 --eval_type test


# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug2_easy0.42_hard0.58_total20000 --eval_type train_aug --num_samples 4 --seed 0
