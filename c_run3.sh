# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_50_unmemorized_eq_50_total5292_epochs10/checkpoint-440 --eval_type test

CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_4_total2646_epochs5/checkpoint-550 --eval_type test
CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_4_total2646_epochs5/checkpoint-441 --eval_type test
CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_4_total2646_epochs5/checkpoint-331 --eval_type test
CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_4_total2646_epochs5/checkpoint-220 --eval_type test
CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_4_total2646_epochs5/checkpoint-110 --eval_type test


CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_4_total2646_epochs5/checkpoint-550 --eval_type train_aug_subsample
CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_4_total2646_epochs5/checkpoint-441 --eval_type train_aug_subsample
CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_4_total2646_epochs5/checkpoint-331 --eval_type train_aug_subsample
CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_4_total2646_epochs5/checkpoint-220 --eval_type train_aug_subsample
CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_4_total2646_epochs5/checkpoint-110 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_50_unmemorized_eq_50_total5292_epochs10/checkpoint-440 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-220 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-220 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-440 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-440 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-2200 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_0_total2646_epochs5/checkpoint-550 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_0_total2646_epochs5/checkpoint-550 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs5/checkpoint-2080 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs5/checkpoint-2080 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs20/checkpoint-8320 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs20/checkpoint-8320 --eval_type train_aug_subsample



# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_memorized_total20000_epochs20/checkpoint-11662 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_memorized_total20000_epochs20/checkpoint-11662 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_memorized_total20000_epochs20/checkpoint-13328 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/math_aug3_memorized_total20000_epochs20/checkpoint-13328 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_20epochs/ --eval_type test
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_20epochs/checkpoint-5598 --eval_type train

# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_20epochs/checkpoint-6220 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_20epochs/checkpoint-6220 --eval_type train


# EASY_RATIO=$(echo "(1 - $HARD_RATIO) * 0.35" | bc -l)
# MEDIUM_RATIO=$(echo "(1 - $HARD_RATIO) * 0.65" | bc -l)

# echo $MEDIUM_RATIO

# torchrun --nproc_per_node=4 --master_port=1234 math_aug2_train.py --num_train_points 20000 --easy_ratio 0.42 --hard_ratio 0.58 --num_epochs 5


# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard5_30000 --eval_type test --seed 3
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard10_30000 --eval_type test --seed 3
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_rand_30000 --eval_type test --seed 3


# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard5_20000 --eval_type test --seed 1
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard5_20000 --eval_type test --seed 3
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard5_20000 --eval_type test --seed 4

# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard10_30000 --eval_type test --seed 2


# # NEED TO DO
# NUM_TRAIN_POINTS=40000
# for EASY_RATIO in 0. 0.25
# do
#     HARD_RATIO=$(echo "(1 - $EASY_RATIO)" | bc -l)

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug2_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --hard_ratio ${HARD_RATIO}
    
# done










# NUM_TRAIN_POINTS=5000
# for EASY_RATIO in 0.75 0.
# do
#     HARD_RATIO=$(echo "(1 - $EASY_RATIO)" | bc -l)

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug2_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --hard_ratio ${HARD_RATIO}
    
# done




# NUM_TRAIN_POINTS=40000
# for HARD_RATIO in 0.5
# do
#     EASY_RATIO=$(echo "(1 - $HARD_RATIO) * 0.35" | bc -l)
#     MEDIUM_RATIO=$(echo "(1 - $HARD_RATIO) * 0.65" | bc -l)

#     echo $MEDIUM_RATIO

#     # torchrun --nproc_per_node=4 --master_port=1234 math_aug_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO}
    
#     # EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     EASY_RATIO=$(echo "scale=2; $EASY_RATIO/1." | bc -l)
#     # MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
#     MEDIUM_RATIO=$(echo "scale=2; $MEDIUM_RATIO/1." | bc -l)
#     # HARD_RATIO=$(printf "%.2f" $HARD_RATIO)
#     HARD_RATIO=$(echo "scale=2; $HARD_RATIO/1." | bc -l)

#     echo $MEDIUM_RATIO
    
#     # ray stop --force
#     # CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}"
#     # CUDA_VISIBLE_DEVICES=0 python math_eval.py --eval_type test --ckpt_dir ${CKPT_NAME} --num_devices 4
#     # ray stop --force
#     # python math_eval.py --eval_type train_aug_subsample --ckpt_dir ${CKPT_NAME} --num_devices 4
# done