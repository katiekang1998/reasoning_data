


RUN_NAME=gsm8k_orig_3epochs_full_lr5e-05_bs120_Qwen-14B

# CUDA_VISIBLE_DEVICES=2,3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type test --num_samples 5
CUDA_VISIBLE_DEVICES=2,3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type train --num_samples 5
CUDA_VISIBLE_DEVICES=2,3 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 124



# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_eighth_lr5e-05_bs128 --ckpt 42 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_quarter_lr5e-05_bs128 --ckpt 84 &
# wait

# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_eighth_lr5e-05_bs128/checkpoint-42 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_eighth_lr5e-05_bs128/checkpoint-42 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_quarter_lr5e-05_bs128/checkpoint-84 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_quarter_lr5e-05_bs128/checkpoint-84 --eval_type train --num_samples 5 &
# wait


# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 2e-4 --epochs 1
# CKPT=58
# RUN_NAME=gsm8k_orig_1epochs_full_lr0.0002_bs128

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
# wait 



# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 2e-4 --epochs 3

# RUN_NAME=gsm8k_orig_3epochs_full_lr0.0002_bs128

# for CKPT in 58 116 174
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done




# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-4 --epochs 6

# RUN_NAME=gsm8k_orig_6epochs_full_lr0.0005_bs128

# for CKPT in 58 116 175 233 292 348
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done


# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-175 --eval_type train_gpt4o --num_samples 50
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-233 --eval_type train_gpt4o --num_samples 5


# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-07_bs128/checkpoint-175 --eval_type train_gpt4o --num_samples 50
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-07_bs128/checkpoint-233 --eval_type train_gpt4o --num_samples 5


# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-05_bs128/checkpoint-175 --eval_type train_gpt4o --num_samples 50
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-05_bs128/checkpoint-233 --eval_type train_gpt4o --num_samples 50


# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-05_bs128_2/checkpoint-175 --eval_type train --num_samples 50
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-05_bs128_2/checkpoint-233 --eval_type train --num_samples 50


# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_12epochs_full_lr5e-07_bs128/checkpoint-232 --eval_type train --num_samples 5
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_12epochs_full_lr5e-07_bs128/checkpoint-232 --eval_type test --num_samples 50




# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-116 --eval_type train --num_samples 50
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-116 --eval_type test --num_samples 50




# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-07_bs128/checkpoint-116 --eval_type train
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-07_bs128/checkpoint-116 --eval_type test

# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-07_bs128/checkpoint-348 --eval_type train
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-07_bs128/checkpoint-348 --eval_type test

# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-623 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-623 --eval_type train

# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-1866 --eval_type test

# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-935 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-935 --eval_type train

# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-1558 --eval_type test


# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-1246 --eval_type test
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-1246 --eval_type train

# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-1558 --eval_type train

# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_2_total2646_epochs5/checkpoint-550 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_2_total2646_epochs5/checkpoint-441 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_2_total2646_epochs5/checkpoint-331 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_2_total2646_epochs5/checkpoint-220 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_2_total2646_epochs5/checkpoint-110 --eval_type test


# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_2_total2646_epochs5/checkpoint-550 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_2_total2646_epochs5/checkpoint-441 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_2_total2646_epochs5/checkpoint-331 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_2_total2646_epochs5/checkpoint-220 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_2_total2646_epochs5/checkpoint-110 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_50_unmemorized_eq_50_total5292_epochs10/checkpoint-1320 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_50_unmemorized_eq_50_total5292_epochs10/checkpoint-1320 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_50_unmemorized_eq_50_total5292_epochs10/checkpoint-1760 --eval_type test

# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-1100 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-1100 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-1320 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-1320 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-1980 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs10/checkpoint-1664 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs5/checkpoint-1250 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs5/checkpoint-1250 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs5/checkpoint-833 --eval_type test


# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs20/checkpoint-4160 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs20/checkpoint-4160 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs20/checkpoint-4992 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs20/checkpoint-4992 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-844 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-844 --eval_type train

# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-1125 --eval_type test
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-1125 --eval_type train

# CUDA_VISIBLE_DEVICES=0 python math_eval.py --eval_type test --ckpt_dir ckpts/math_aug2_easy0.42_hard0.58_total20000
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --eval_type test --ckpt_dir ckpts/math_aug2_unmemorized_total12159
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --eval_type test --ckpt_dir ckpts/math_aug2_rand_total12159



# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard5_30000 --eval_type test --seed 1
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard10_30000 --eval_type test --seed 1
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_rand_30000 --eval_type test --seed 1

# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard15_20000 --eval_type test --seed 1
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard15_20000 --eval_type test --seed 3
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard15_20000 --eval_type test --seed 4

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_aug_llama2.py --num_train_points 20000 --data_type hard15
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_aug_llama2.py --num_train_points 30000 --data_type rand


# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_aug_llama2.py --num_train_points 5000 --data_type hard
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_aug_llama2.py --num_train_points 5000 --data_type rand



# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_all_2epochs_llama2 --eval_type train_aug_2 --seed 2
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_all_2epochs_llama2 --eval_type test --seed 1

# torchrun --nproc_per_node=4 --master_port=1234 math_all_train_llama2.py
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_all_train_llama2.py


# NUM_TRAIN_POINTS=20000
# for EASY_RATIO in 0. 0.25
# do
#     HARD_RATIO=$(echo "(1 - $EASY_RATIO)" | bc -l)

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug2_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --hard_ratio ${HARD_RATIO} --num_epochs 5
    
# done


# NUM_TRAIN_POINTS=40000
# for EASY_RATIO in 0.75
# do
#     HARD_RATIO=$(echo "(1 - $EASY_RATIO)" | bc -l)

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug2_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --hard_ratio ${HARD_RATIO}
    
# done


# NUM_TRAIN_POINTS=20000
# for HARD_RATIO in 0.75 1.
# do
#     EASY_RATIO=$(echo "(1 - $HARD_RATIO) * 0.35" | bc -l)
#     MEDIUM_RATIO=$(echo "(1 - $HARD_RATIO) * 0.65" | bc -l)

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO}
    
#     # EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     # MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
#     # HARD_RATIO=$(printf "%.2f" $HARD_RATIO)
#     # EASY_RATIO=$(echo "scale=2; $EASY_RATIO/1" | bc -l)
#     # MEDIUM_RATIO=$(echo "scale=2; $MEDIUM_RATIO/1" | bc -l)
#     # HARD_RATIO=$(echo "scale=2; $HARD_RATIO/1" | bc -l)
    
#     # ray stop --force
#     # CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}"
#     # python math_eval.py --eval_type test --ckpt_dir ${CKPT_NAME} --num_devices 4
#     # ray stop --force
#     # python math_eval.py --eval_type train_aug_subsample --ckpt_dir ${CKPT_NAME} --num_devices 4
# done


# NUM_TRAIN_POINTS=10000


# for HARD_RATIO in 0.75
# do
#     EASY_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)
#     MEDIUM_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)

#     TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=4 --master_port=1235 math_aug_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO}
    
#     EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
#     HARD_RATIO=$(printf "%.2f" $HARD_RATIO)

#     # ray stop
#     # CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}"
#     # CUDA_VISIBLE_DEVICES=0,1 python math_eval.py --eval_type train_aug_subsample --ckpt_dir ${CKPT_NAME} &
#     # P1=$!
#     # CUDA_VISIBLE_DEVICES=2,3 python math_eval.py --eval_type test --ckpt_dir ${CKPT_NAME} &
#     # P2=$!
#     # wait $P1 $P2
# done



# NUM_TRAIN_POINTS=20000
# for HARD_RATIO in 1. 0.75 0.5
# do
#     EASY_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)
#     MEDIUM_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)

#     TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=4 --master_port=1235 math_aug_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO}
    
#     EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
#     HARD_RATIO=$(printf "%.2f" $HARD_RATIO)

#     # ray stop
#     # CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}"
#     # CUDA_VISIBLE_DEVICES=0,1 python math_eval.py --eval_type train_aug_subsample --ckpt_dir ${CKPT_NAME} &
#     # P1=$!
#     # CUDA_VISIBLE_DEVICES=2,3 python math_eval.py --eval_type test --ckpt_dir ${CKPT_NAME} &
#     # P2=$!
#     # wait $P1 $P2
# done