
# torchrun --nproc_per_node=4 --master_port=1234 math_train.py --dont_save_intermediate --learning_rate 5e-5 --epochs 6 --train_type full2
RUN_NAME=math_orig_6epochs_full2_lr5e-05_bs128
CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/$RUN_NAME --eval_type train_we --num_samples 10 &
CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME --eval_type train_first --num_samples 10 &
CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME --eval_type train --num_samples 10 &
wait


# RUN_NAME=gsm8k_orig_3epochs_full2_lr2e-05_bs128
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-58 --temp 0 --num_samples 1 --eval_type train &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-116 --temp 0 --num_samples 1 --eval_type train &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-174 --temp 0 --num_samples 1 --eval_type train &
# wait



# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_weight_norm.py --ckpt_dir $RUN_NAME --ckpt 58
# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_weight_norm.py --ckpt_dir $RUN_NAME --ckpt 116
# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_weight_norm.py --ckpt_dir $RUN_NAME --ckpt 174

# RUN_NAME=gsm8k_orig_3epochs_full2_lr5e-07_bs128
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-58 --temp 0 --num_samples 1 --eval_type train &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-116 --temp 0 --num_samples 1 --eval_type train &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-174 --temp 0 --num_samples 1 --eval_type train &
# wait


# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_weight_norm.py --ckpt_dir $RUN_NAME --ckpt 58
# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_weight_norm.py --ckpt_dir $RUN_NAME --ckpt 116
# # CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_weight_norm.py --ckpt_dir $RUN_NAME --ckpt 174

# RUN_NAME=gsm8k_orig_3epochs_full2_lr0.0002_bs128
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-58 --temp 0 --num_samples 1 --eval_type train &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-116 --temp 0 --num_samples 1 --eval_type train &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-174 --temp 0 --num_samples 1 --eval_type train &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_weight_norm.py --ckpt_dir $RUN_NAME --ckpt 58
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_weight_norm.py --ckpt_dir $RUN_NAME --ckpt 116
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_weight_norm.py --ckpt_dir $RUN_NAME --ckpt 174

# torchrun --nproc_per_node=4 --master_port=1234 math_train2.py --train_type 3copies

# RUN_NAME=math_amrith_3epochs_3copies_lr2e-05_bs24
# for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
# do
#     # Extract the checkpoint number from the folder name
#     CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
#     echo $CKPT
#     CUDA_VISIBLE_DEVICES=0 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 10 &
#     CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5  --seed 3 &

#     wait
# done

# torchrun --nproc_per_node=4 --master_port=1234 math_train2.py --train_type 7copies

# RUN_NAME=math_amrith_3epochs_7copies_lr2e-05_bs24
# for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
# do
#     # Extract the checkpoint number from the folder name
#     CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
#     echo $CKPT
#     CUDA_VISIBLE_DEVICES=0 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 10 &
#     CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5  --seed 3 &

#     wait
# done



# torchrun --nproc_per_node=4 --master_port=1234 math_train2.py --train_type 15copies

# RUN_NAME=math_amrith_3epochs_15copies_lr2e-05_bs24
# for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
# do
#     # Extract the checkpoint number from the folder name
#     CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
#     echo $CKPT
#     CUDA_VISIBLE_DEVICES=0 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 10 &
#     CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5  --seed 3 &

#     wait
# done



# torchrun --nproc_per_node=4 --master_port=1234 math_train2.py --train_type 11copies

# RUN_NAME=math_amrith_3epochs_11copies_lr2e-05_bs24
# for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
# do
#     # Extract the checkpoint number from the folder name
#     CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
#     echo $CKPT
#     CUDA_VISIBLE_DEVICES=0 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 10 &
#     CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5  --seed 3 &

#     wait
# done



# torchrun --nproc_per_node=4 --master_port=1234 math_train2.py --train_type 19copies

# RUN_NAME=math_amrith_3epochs_19copies_lr2e-05_bs24
# for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
# do
#     # Extract the checkpoint number from the folder name
#     CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
#     echo $CKPT
#     CUDA_VISIBLE_DEVICES=0 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 10 &
#     CUDA_VISIBLE_DEVICES=3 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5  --seed 3 &

#     wait
# done

# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-117 --eval_type test --num_samples 5 --seed 3 &
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-234 --eval_type test --num_samples 5 --seed 3 &
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-351 --eval_type test --num_samples 5 --seed 3 &
# wait

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --train_type full2 --learning_rate 5e-5
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --train_type full2 --learning_rate 2e-4
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --train_type full2 --learning_rate 5e-7


# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train10.py --train_type 7copies_threshold0.75_4newcopies
# RUN_NAME=gsm8k_amrith_3epochs_7copies_threshold0.75_4newcopies_lr2e-05_bs128
# for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
# do
#     # Extract the checkpoint number from the folder name
#     CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
#     echo $CKPT
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 10 &
#     wait
# done

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train10.py --train_type 0copies_threshold0.75_1newcopies
# RUN_NAME=gsm8k_amrith_3epochs_0copies_threshold0.75_1newcopies_lr2e-05_bs128
# for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
# do
#     # Extract the checkpoint number from the folder name
#     CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
#     echo $CKPT
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 10 &
#     wait
# done

# RUN_NAME=gsm8k_amrith_3epochs_0copies_threshold0.75_0.5newcopies_lr2e-05_bs128
# for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
# do
#     # Extract the checkpoint number from the folder name
#     CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
#     echo $CKPT
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 10 &
#     wait
# done


# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-292 --eval_type train_gpt4o --num_samples 50

# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-07_bs128/checkpoint-292 --eval_type train_gpt4o --num_samples 50


# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-05_bs128/checkpoint-292 --eval_type train_gpt4o --num_samples 50



# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-05_bs128_2/checkpoint-292 --eval_type train --num_samples 50


# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_12epochs_full_lr5e-07_bs128/checkpoint-348 --eval_type train --num_samples 5
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_12epochs_full_lr5e-07_bs128/checkpoint-348 --eval_type test --num_samples 50

# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_12epochs_full_lr5e-07_bs128/checkpoint-464 --eval_type train --num_samples 5
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_12epochs_full_lr5e-07_bs128/checkpoint-464 --eval_type test --num_samples 50


# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-175 --eval_type train --num_samples 50
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-175 --eval_type test --num_samples 50


# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-233 --eval_type train --num_samples 50
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-233 --eval_type test --num_samples 50


# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-07_bs128/checkpoint-175 --eval_type train
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-07_bs128/checkpoint-175 --eval_type test

# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-935 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-935 --eval_type train

# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-05/checkpoint-1558 --eval_type test


# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_3_total2646_epochs5/checkpoint-550 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_3_total2646_epochs5/checkpoint-441 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_3_total2646_epochs5/checkpoint-331 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_3_total2646_epochs5/checkpoint-220 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_3_total2646_epochs5/checkpoint-110 --eval_type test


# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_3_total2646_epochs5/checkpoint-550 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_3_total2646_epochs5/checkpoint-441 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_3_total2646_epochs5/checkpoint-331 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_3_total2646_epochs5/checkpoint-220 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_3_total2646_epochs5/checkpoint-110 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_50_unmemorized_eq_50_total5292_epochs10/checkpoint-880 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand_50_unmemorized_eq_50_total5292_epochs10/checkpoint-880 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-660 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-660 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-880 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-880 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_rand2646_total2646_epochs20/checkpoint-2200 --eval_type test

# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs10/checkpoint-1664 --eval_type test


# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs5/checkpoint-1666 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_leq_3_total10000_epochs5/checkpoint-1666 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs5/checkpoint-833 --eval_type test


# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs20/checkpoint-6656 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_geq_3_total10000_epochs20/checkpoint-6656 --eval_type train_aug_subsample


# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_memorized_total20000_epochs20/checkpoint-8330 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_memorized_total20000_epochs20/checkpoint-8330 --eval_type train_aug_subsample

# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_memorized_total20000_epochs20/checkpoint-9996 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_aug3_memorized_total20000_epochs20/checkpoint-9996 --eval_type train_aug_subsample



# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-1406 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-1406 --eval_type train

# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-1686 --eval_type test
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_unmemorized/checkpoint-1686 --eval_type train


# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard5_30000 --eval_type test --seed 4
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard10_30000 --eval_type test --seed 4
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_rand_30000 --eval_type test --seed 4

# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard10_20000 --eval_type test --seed 1
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard10_20000 --eval_type test --seed 3
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_aug_llama2_hard10_20000 --eval_type test --seed 4

# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_all_2epochs_llama2 --eval_type train_aug_1 --seed 3
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_all_2epochs_llama2 --eval_type test --seed 2

# NUM_TRAIN_POINTS=20000
# for EASY_RATIO in 0.
# do
#     HARD_RATIO=$(echo "(1 - $EASY_RATIO)" | bc -l)

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug2_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --hard_ratio ${HARD_RATIO} --save_intermediate True
    
# done

# NUM_TRAIN_POINTS=40000
# for EASY_RATIO in 0.5
# do
#     HARD_RATIO=$(echo "(1 - $EASY_RATIO)" | bc -l)

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug2_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --hard_ratio ${HARD_RATIO}
    
# done


# NUM_TRAIN_POINTS=10000
# for HARD_RATIO in 1. 0. 0.5 0.25 0.75
# do
#     EASY_RATIO=$(echo "(1 - $HARD_RATIO)" | bc -l)

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug2_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --hard_ratio ${HARD_RATIO}
    
#     EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     HARD_RATIO=$(printf "%.2f" $HARD_RATIO)
    
#     ray stop --force
#     CKPT_NAME="ckpts/math_aug2_easy${EASY_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}"
#     python math_eval.py --eval_type test --ckpt_dir ${CKPT_NAME} --num_devices 4
#     # ray stop --force
#     # python math_eval.py --eval_type train_aug_subsample --ckpt_dir ${CKPT_NAME} --num_devices 4
# done

# NUM_TRAIN_POINTS=20000
# for HARD_RATIO in 0.5 0.25
# do
#     EASY_RATIO=$(echo "(1 - $HARD_RATIO) * 0.35" | bc -l)
#     MEDIUM_RATIO=$(echo "(1 - $HARD_RATIO) * 0.65" | bc -l)

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO}
    
#     # EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     # MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
#     # HARD_RATIO=$(printf "%.2f" $HARD_RATIO)
    
#     # ray stop --force
#     # CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}"
#     # python math_eval.py --eval_type test --ckpt_dir ${CKPT_NAME} --num_devices 4
#     # ray stop --force
#     # CUDA_VISIBLE_DEVICES=0 python math_eval.py --eval_type train_aug_subsample --ckpt_dir ${CKPT_NAME} --num_devices 4
# done

# NUM_TRAIN_POINTS=10000
# for HARD_RATIO in 0. 0.25 0.5 0.75
# do
#     EASY_RATIO=$(echo "(1 - $HARD_RATIO) * 0.35" | bc -l)
#     MEDIUM_RATIO=$(echo "(1 - $HARD_RATIO) * 0.65" | bc -l)

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO}
    
#     EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
#     HARD_RATIO=$(printf "%.2f" $HARD_RATIO)
    
#     ray stop --force
#     CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}"
#     python math_eval.py --eval_type test --ckpt_dir ${CKPT_NAME} --num_devices 4
#     ray stop --force
#     python math_eval.py --eval_type train_aug_subsample --ckpt_dir ${CKPT_NAME} --num_devices 4
# done

# NUM_TRAIN_POINTS=20000
# for HARD_RATIO in 0. 0.25
# do
#     EASY_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)
#     MEDIUM_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)

#     # torchrun --nproc_per_node=4 --master_port=1234 math_aug_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO}
    
#     EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
#     HARD_RATIO=$(printf "%.2f" $HARD_RATIO)

#     ray stop --force

#     CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}"
#     python math_eval.py --eval_type test --ckpt_dir ${CKPT_NAME} --num_devices 4
# done


# NUM_TRAIN_POINTS=10000
# for HARD_RATIO in 0. 0.25 0.5 1.
# do
#     EASY_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)
#     MEDIUM_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)

#     # torchrun --nproc_per_node=4 --master_port=1234 math_aug_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO}
    
#     EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
#     HARD_RATIO=$(printf "%.2f" $HARD_RATIO)
    
#     ray stop --force
#     CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}"
#     python math_eval.py --eval_type train_aug_subsample --ckpt_dir ${CKPT_NAME} --num_devices 4
# done



# NUM_TRAIN_POINTS=20000
# for HARD_RATIO in 0. 0.25
# do
#     EASY_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)
#     MEDIUM_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)

#     # torchrun --nproc_per_node=4 --master_port=1234 math_aug_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO}
    
#     EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
#     HARD_RATIO=$(printf "%.2f" $HARD_RATIO)

#     ray stop --force

#     CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}"
#     python math_eval.py --eval_type train_aug_subsample --ckpt_dir ${CKPT_NAME} --num_devices 4
# done


# NUM_TRAIN_POINTS=5000

# for NUM_EPOCHS in 2 5 10
# do
#     HARD_RATIO=0.34
#     EASY_RATIO=0.23
#     MEDIUM_RATIO=0.43

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug_train.py --num_train_points 5000 --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO} --num_epochs ${NUM_EPOCHS}
    
#     EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
#     HARD_RATIO=$(printf "%.2f" $HARD_RATIO)
    
#     ray stop --force
#     CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs"
#     python math_eval.py --eval_type test --ckpt_dir ${CKPT_NAME} --num_devices 4

#     HARD_RATIO=1.
#     EASY_RATIO=0.
#     MEDIUM_RATIO=0.

#     torchrun --nproc_per_node=4 --master_port=1234 math_aug_train.py --num_train_points 5000 --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO} --num_epochs ${NUM_EPOCHS}
    
#     EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
#     MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
#     HARD_RATIO=$(printf "%.2f" $HARD_RATIO)
    
#     ray stop --force
#     CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}_${NUM_EPOCHS}epochs"
#     python math_eval.py --eval_type test --ckpt_dir ${CKPT_NAME} --num_devices 4
# done