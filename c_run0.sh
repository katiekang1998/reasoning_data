


# 6 new copies
# 10 new copies


# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train10.py --train_type 1copies_threshold0.75_6newcopies
# RUN_NAME=gsm8k_amrith_3epochs_1copies_threshold0.75_6newcopies_lr2e-05_bs128
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

torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train10.py --train_type 1copies_threshold0.75_8newcopies
RUN_NAME=gsm8k_amrith_3epochs_1copies_threshold0.75_8newcopies_lr2e-05_bs128
for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
do
    # Extract the checkpoint number from the folder name
    CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
    echo $CKPT
    CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
    CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
    CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 10 &
    wait
done


torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train10.py --train_type 1copies_threshold0.75_4newcopies
RUN_NAME=gsm8k_amrith_3epochs_1copies_threshold0.75_4newcopies_lr2e-05_bs128
for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
do
    # Extract the checkpoint number from the folder name
    CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
    echo $CKPT
    CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
    CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
    CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 10 &
    wait
done

# RUN_NAME=gsm8k_amrith_3epochs_3copies_threshold0.5_lr2e-05_bs128
# for CKPT in 350 700 1050
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     wait
# done

# torchrun --nproc_per_node=4 --master_port=1234 math_train.py --train_type full --learning_rate 2e-5 --epochs 3
# RUN_NAME=math_orig_3epochs_full_lr2e-05_bs128
# for CKPT in 312 625 936
# do
#     CUDA_VISIBLE_DEVICES=0 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     wait
# done

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train7.py --train_type memorized_half --dont_save_intermediate
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train7.py --train_type unmemorized_half --dont_save_intermediate
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train7.py --train_type memorized_3quarter --dont_save_intermediate
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train7.py --train_type unmemorized_3quarter --dont_save_intermediate


# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train6.py --train_type batch_1_threshold0.25
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train6.py --train_type batch_1_threshold1
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train6.py --train_type batch_1_threshold0.50
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train6.py --train_type batch_1_threshold0.75


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_amrith_3epochs_batch_1_threshold0.25_lr2e-05_bs128/checkpoint-1881 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_amrith_3epochs_batch_1_threshold1_lr2e-05_bs128/checkpoint-1881 --eval_type test --num_samples 5 &
# wait
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 2e-4 --train_type shuffle5

# RUN_NAME=gsm8k_amrith_3epochs_batch_1_all_lr2e-05_bs128

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-408 --eval_type train --num_samples 50 --seed 2 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-817 --eval_type train --num_samples 50 --seed 2 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-1224 --eval_type train --num_samples 50 --seed 2 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-408 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-817 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-1224 --eval_type test --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 408 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 817 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 1224 &
# wait

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 2e-5 --train_type shuffle5

# RUN_NAME=gsm8k_orig_3epochs_shuffle5_lr2e-05_bs128

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-58 --eval_type train --num_samples 50 --seed 2 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-116 --eval_type train --num_samples 50 --seed 2 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-174 --eval_type train --num_samples 50 --seed 2 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-58 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-116 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-174 --eval_type test --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 58 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 116 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 174 &
# wait


# for CKPT in 58 116 174
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_3epochs_shuffle1_lr0.0002_bs128/checkpoint-$CKPT --eval_type train --num_samples 50 --seed 0 &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_3epochs_shuffle1_lr0.0002_bs128/checkpoint-$CKPT --eval_type train --num_samples 50 --seed 1 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_3epochs_shuffle1_lr0.0002_bs128/checkpoint-$CKPT --eval_type train --num_samples 50 --seed 2 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_3epochs_shuffle1_lr0.0002_bs128/checkpoint-$CKPT --eval_type train --num_samples 50 --seed 3 &
#     wait
# done

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_3epochs_shuffle1_lr0.0002_bs128/checkpoint-58 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_3epochs_shuffle1_lr0.0002_bs128/checkpoint-116 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_3epochs_shuffle1_lr0.0002_bs128/checkpoint-174 --eval_type test --num_samples 5 &
# wait


# RUN_NAME=gsm8k_orig_3epochs_shuffle1_lr0.0002_bs128
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 58 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 116 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 174 &
# wait


# # torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train5.py --train_type threshold1
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train5.py --train_type threshold0.125_2
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train5.py --train_type threshold0.25_2
# # torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train5.py --train_type threshold0.25rand
# # torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train5.py --train_type threshold0.5
# # torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train5.py --train_type threshold0.5rand
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train5.py --train_type threshold0.75

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train5.py --train_type threshold0.25 --learning_rate 2e-5
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train5.py --train_type threshold0.25rand  --learning_rate 2e-5

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_amrith_3epochs_threshold0.25rand_lr2e-05_bs128 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_amrith_3epochs_threshold0.25_lr2e-05_bs128 --eval_type test --num_samples 5 &
# # CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_amrith_3epochs_threshold0.75_lr5e-05_bs128 --eval_type test --num_samples 5 &
# # CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_amrith_6epochs_threshold0.25_lr5e-05_bs128 --eval_type test --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_amrith_6epochs_threshold0.25rand_lr5e-05_bs128 --eval_type test --num_samples 5

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train4.py --train_type add_rand_quarter
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train4.py --train_type add_rand_quarter
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train4.py --train_type add_rand_quarter
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train4.py --train_type add_hard_quarter
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train4.py --train_type add_easymedium_quarter

# for TRAIN_TYPE in add_rand_quarter add_hard_quarter add_easy_quarter add_medium_quarter add_easymedium_quarter
# do
#     RUN_NAME=gsm8k_gpt4o_3epochs_${TRAIN_TYPE}_lr5e-05_bs128
#     echo $RUN_NAME
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-73 --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-146 --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-219 --eval_type test --num_samples 5 &
#     wait


#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 73 &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 146 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 219 &
#     wait

#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-73 --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-146 --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-219 --eval_type train --num_samples 5 &
#     wait
# done

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train4.py --train_type add_hardmedium_quarter

# for TRAIN_TYPE in add_hardmedium_quarter
# do
#     RUN_NAME=gsm8k_gpt4o_3epochs_${TRAIN_TYPE}_lr5e-05_bs128
#     echo $RUN_NAME
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-73 --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-146 --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-219 --eval_type test --num_samples 5 &
#     wait

#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 73 &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 146 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 219 &
#     wait


#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-73 --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-146 --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-219 --eval_type train --num_samples 5 &
#     wait
# done

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train4.py --train_type add_unmemorized
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train4.py --train_type add_full
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train4.py --train_type add_rand


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_gpt4o_3epochs_add_memorized_lr5e-05_bs128/checkpoint-261 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_gpt4o_3epochs_add_unmemorized_lr5e-05_bs128/checkpoint-261 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_gpt4o_3epochs_add_full_lr5e-05_bs128/checkpoint-348 --eval_type test --num_samples 5 &
# wait


# python eval_mmlu.py --ckpt_dir ckpts/gsm8k_orig_6epochs_shuffle1_lr5e-05_bs128/checkpoint-58
# python eval_mmlu.py --ckpt_dir ckpts/gsm8k_orig_6epochs_shuffle1_lr5e-05_bs128/checkpoint-116
# python eval_mmlu.py --ckpt_dir ckpts/gsm8k_orig_6epochs_shuffle1_lr5e-05_bs128/checkpoint-175
# python eval_mmlu.py --ckpt_dir ckpts/gsm8k_orig_6epochs_shuffle1_lr5e-05_bs128/checkpoint-233
# python eval_mmlu.py --ckpt_dir ckpts/gsm8k_orig_6epochs_shuffle1_lr5e-05_bs128/checkpoint-348

# python eval_mmlu.py --ckpt_dir ckpts/gsm8k_orig_6epochs_shuffle1_lr5e-07_bs128/checkpoint-348
# python eval_mmlu.py --ckpt_dir ckpts/gsm8k_orig_1epochs_shuffle1_lr5e-05_bs128/checkpoint-58
# python eval_mmlu.py --ckpt_dir ckpts/gsm8k_orig_24epochs_eighth_lr5e-05_bs128/checkpoint-168



# CUDA_VISIBLE_DEVICES=0 python eval_mmlu_perplexity.py --ckpt_dir gsm8k_orig_24epochs_eighth_lr5e-05_bs128 --ckpt 168 &
# CUDA_VISIBLE_DEVICES=1 python eval_mmlu_perplexity.py --ckpt_dir gsm8k_orig_6epochs_shuffle1_lr5e-05_bs128 --ckpt 348 &
# CUDA_VISIBLE_DEVICES=2 python eval_mmlu_perplexity.py --ckpt_dir gsm8k_orig_6epochs_shuffle1_lr5e-07_bs128 --ckpt 348 &
# CUDA_VISIBLE_DEVICES=3 python eval_mmlu_perplexity.py --ckpt_dir gsm8k_orig_1epochs_shuffle1_lr5e-05_bs128 --ckpt 58 &
# wait





# for CKPT in 168 140 112 84 56 28
# do
#     python eval_mmlu.py --ckpt_dir ckpts/gsm8k_orig_24epochs_eighth_lr5e-05_bs128/checkpoint-$CKPT
# done

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --train_type shuffle1
# RUN_NAME=gsm8k_orig_6epochs_shuffle1_lr5e-05_bs128
# for CKPT in 58 116 175 233 292 348
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --train_type shuffle2
# RUN_NAME=gsm8k_orig_6epochs_shuffle2_lr5e-05_bs128
# for CKPT in 58 116 175 233 292 348
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --train_type shuffle1 --learning_rate 5e-7

# RUN_NAME=gsm8k_orig_3epochs_gpt4o_memorized_lr5e-05_bs128
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-58 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-116 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-174 --eval_type train --num_samples 5 &
# wait


# RUN_NAME=gsm8k_orig_3epochs_gpt4o_unmemorized_lr5e-05_bs128
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-58 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-116 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-174 --eval_type train --num_samples 5 &
# wait

# RUN_NAME=math_orig_3epochs_subsample_unmemorized_lr5e-05_bs128
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-175 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-351 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-348 --eval_type test --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_orig_3epochs_subsample_unmemorized_lr5e-05_bs128/checkpoint-525 --eval_type test --num_samples 5

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-58 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-116 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-175 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-233 --eval_type test --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-58 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-116 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-175 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-233 --eval_type train --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-292 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-348 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 292 &
# CUDA_VISIBLE_DEVICES=3 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 348 &
# wait

# CUDA_VISIBLE_DEVICES=0 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 58 &
# CUDA_VISIBLE_DEVICES=1 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 116 &
# CUDA_VISIBLE_DEVICES=2 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 175 &
# CUDA_VISIBLE_DEVICES=3 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 233 &
# wait


# torchrun --nproc_per_node=4 --master_port=1234 math_train.py --epochs 1 --learning_rate 5e-7
# RUN_NAME=math_orig_1epochs_full_lr5e-07_bs128
# for CKPT in 312
# do
#     CUDA_VISIBLE_DEVICES=0 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     wait 
# done


# torchrun --nproc_per_node=4 --master_port=1234 math_train.py --epochs 3 --learning_rate 5e-7
# RUN_NAME=math_orig_3epochs_full_lr5e-07_bs128
# for CKPT in 312 625 936
# do
#     CUDA_VISIBLE_DEVICES=0 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     wait 
# done


# torchrun --nproc_per_node=4 --master_port=1234 math_train.py --epochs 6 --learning_rate 5e-7
# RUN_NAME=math_orig_6epochs_full_lr5e-07_bs128
# for CKPT in 312 625 937 1250 1562 1872
# do
#     CUDA_VISIBLE_DEVICES=0 python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     wait 
# done


# RUN_NAME=gsm8k_orig_3epochs_add_memorized_lr5e-05_bs128

# for CKPT in 83 167 249
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train2.py --train_type add_rand
# RUN_NAME=gsm8k_orig_3epochs_add_rand_lr5e-05_bs128
# for CKPT in 83 167 249
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train2.py --train_type add_memorized_orig
# RUN_NAME=gsm8k_orig_3epochs_add_memorized_orig_lr5e-05_bs128
# for CKPT in 83 167 249
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done


# RUN_NAME=gsm8k_orig_6epochs_full_lr5e-05_bs120_Qwen-14B

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-62 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 62 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-62 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=4 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 124 &
# CUDA_VISIBLE_DEVICES=5 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type test --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-186 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 186 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-186 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-249 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=4 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 249 &
# CUDA_VISIBLE_DEVICES=5 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-249 --eval_type test --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-311 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 311 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-311 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-372 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=4 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 372 &
# CUDA_VISIBLE_DEVICES=5 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-372 --eval_type test --num_samples 5 &
# wait

# delete_ckpt $RUN_NAME


# torchrun --nproc_per_node=6 --master_port=1234 gsm8k_train.py --learning_rate 5e-7 --epochs 3 --model Qwen/Qwen-14B --batch_size 120

# RUN_NAME=gsm8k_orig_3epochs_full_lr5e-07_bs120_Qwen-14B


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-62 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-186 --eval_type test --num_samples 5 &
# wait


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-62 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 62 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 124 &
# CUDA_VISIBLE_DEVICES=4 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-186 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=5 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 186 &
# wait

# delete_ckpt $RUN_NAME




# torchrun --nproc_per_node=6 --master_port=1234 gsm8k_train.py --learning_rate 2e-4 --epochs 3 --model Qwen/Qwen-14B --batch_size 120

# RUN_NAME=gsm8k_orig_3epochs_full_lr0.0002_bs120_Qwen-14B


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-62 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-186 --eval_type test --num_samples 5 &
# wait


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-62 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 62 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 124 &
# CUDA_VISIBLE_DEVICES=4 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-186 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=5 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 186 &
# wait

# delete_ckpt $RUN_NAME




# torchrun --nproc_per_node=6 --master_port=1234 gsm8k_train.py --learning_rate 5e-7 --epochs 6 --model Qwen/Qwen-14B --batch_size 120
# RUN_NAME=gsm8k_orig_6epochs_full_lr5e-07_bs120_Qwen-14B

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-62 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 62 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-62 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=4 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 124 &
# CUDA_VISIBLE_DEVICES=5 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type test --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-186 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 186 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-186 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-249 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=4 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 249 &
# CUDA_VISIBLE_DEVICES=5 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-249 --eval_type test --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-311 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 311 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-311 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-372 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=4 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 372 &
# CUDA_VISIBLE_DEVICES=5 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-372 --eval_type test --num_samples 5 &
# wait

# delete_ckpt $RUN_NAME


# torchrun --nproc_per_node=6 --master_port=1234 gsm8k_train.py --learning_rate 2e-4 --epochs 6 --model Qwen/Qwen-14B --batch_size 120
# RUN_NAME=gsm8k_orig_6epochs_full_lr0.0002_bs120_Qwen-14B

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-62 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 62 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-62 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=4 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 124 &
# CUDA_VISIBLE_DEVICES=5 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-124 --eval_type test --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-186 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 186 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-186 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-249 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=4 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 249 &
# CUDA_VISIBLE_DEVICES=5 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-249 --eval_type test --num_samples 5 &
# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-311 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 311 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-311 --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-372 --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=4 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt 372 &
# CUDA_VISIBLE_DEVICES=5 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-372 --eval_type test --num_samples 5 &
# wait

# delete_ckpt $RUN_NAME


# RUN_NAME=gsm8k_orig_6epochs_half_lr5e-05_bs128

# for CKPT in 29 58 87 116 145 174
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done

# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-5 --epochs 3 --lora True

# RUN_NAME=gsm8k_orig_6epochs_full_lr5e-05_bs128_lora

# for CKPT in 58 116 174
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done


# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-5 --epochs 6 --train_type half
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-5 --epochs 6 --train_type quarter
# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-5 --epochs 6 --train_type eighth

# CKPT=174

# CUDA_VISIBLE_DEVICES=1  gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_half_lr5e-05_bs128 --ckpt $CKPT &

# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_half_lr5e-05_bs128/checkpoint-$CKPT --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_half_lr5e-05_bs128/checkpoint-$CKPT --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_quarter_lr5e-05_bs128/checkpoint-$CKPT --eval_type test --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_quarter_lr5e-05_bs128/checkpoint-$CKPT --eval_type train --num_samples 5 &
# wait

# # torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-5 --epochs 3 --train_type half

# RUN_NAME=gsm8k_orig_3epochs_half_lr5e-05_bs128

# for CKPT in 29 87
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done


# # torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-7 --epochs 3 --train_type half

# RUN_NAME=gsm8k_orig_3epochs_half_lr5e-07_bs128

# for CKPT in 29 87
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done


# # torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 2e-4 --epochs 3 --train_type half

# RUN_NAME=gsm8k_orig_3epochs_half_lr0.0002_bs128

# for CKPT in 29 87
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done


# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 2e-4 --epochs 3 --lora True

# RUN_NAME=gsm8k_orig_6epochs_full_lr0.0002_bs128_lora

# for CKPT in 58 116 174
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done

# CKPT=58
# RUN_NAME=gsm8k_orig_1epochs_full_lr5e-05_bs128


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
# wait 


# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-4 --epochs 1
# CKPT=58
# RUN_NAME=gsm8k_orig_1epochs_full_lr0.0005_bs128

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
# wait 


# # torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-7 --epochs 1
# CKPT=58
# RUN_NAME=gsm8k_orig_1epochs_full_lr5e-07_bs128

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
# wait 



# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-5 --epochs 3

# RUN_NAME=gsm8k_orig_3epochs_full_lr5e-05_bs128

# for CKPT in 58 116 174
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done


# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-4 --epochs 3

# RUN_NAME=gsm8k_orig_3epochs_full_lr0.0005_bs128

# for CKPT in 58 116 174
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done


# torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-7 --epochs 3

# RUN_NAME=gsm8k_orig_3epochs_full_lr5e-07_bs128

# for CKPT in 58 116 174
# do
#     CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
#     CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
#     CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
#     wait 
# done


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_12epochs_full_lr5e-07_bs128 --ckpt 116 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_12epochs_full_lr5e-07_bs128 --ckpt 232 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_12epochs_full_lr5e-07_bs128 --ckpt 348 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_12epochs_full_lr5e-07_bs128 --ckpt 464 &

# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_12epochs_full_lr5e-07_bs128 --ckpt 580 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_12epochs_full_lr5e-07_bs128 --ckpt 696 &

# wait

# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr5e-07_bs128 --ckpt 116 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr5e-07_bs128 --ckpt 58 &

# wait 

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr5e-07_bs128 --ckpt 348 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr5e-07_bs128 --ckpt 292 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr5e-07_bs128 --ckpt 233 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr5e-07_bs128 --ckpt 175 &

# wait 

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr0.0002_bs128 --ckpt 348 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr0.0002_bs128 --ckpt 292 &
# CUDA_VISIBLE_DEVICES=2 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr0.0002_bs128 --ckpt 233 &
# CUDA_VISIBLE_DEVICES=3 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr0.0002_bs128 --ckpt 175 &

# wait

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr0.0002_bs128 --ckpt 116 &
# CUDA_VISIBLE_DEVICES=1 python gsm8k_eval_perplexity.py --ckpt_dir gsm8k_orig_6epochs_full_lr0.0002_bs128 --ckpt 58 &

# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-58 --eval_type train_gpt4o --num_samples 50
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-116 --eval_type train_gpt4o --num_samples 50


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-07_bs128/checkpoint-58 --eval_type train_gpt4o --num_samples 50
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr5e-07_bs128/checkpoint-116 --eval_type train_gpt4o --num_samples 50


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-05_bs128/checkpoint-58 --eval_type train_gpt4o --num_samples 50
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-05_bs128/checkpoint-116 --eval_type train_gpt4o --num_samples 50


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-05_bs128_2/checkpoint-58 --eval_type train --num_samples 50
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-05_bs128_2/checkpoint-116 --eval_type train --num_samples 50



# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_12epochs_full_lr5e-07_bs128/checkpoint-116 --eval_type train --num_samples 5
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_12epochs_full_lr5e-07_bs128/checkpoint-116 --eval_type test --num_samples 50


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-58 --eval_type train --num_samples 50
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr0.0002_bs128/checkpoint-58 --eval_type test --num_samples 50



# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-07_bs128/checkpoint-58 --eval_type train
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-07_bs128/checkpoint-58 --eval_type test


# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-07_bs128/checkpoint-233 --eval_type train
# CUDA_VISIBLE_DEVICES=0 python gsm8k_eval.py --ckpt_dir ckpts/gsm8k_orig_6epochs_full_lr2e-07_bs128/checkpoint-233 --eval_type test



# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-550 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-441 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-331 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-220 --eval_type test
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-110 --eval_type test


# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-550 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-441 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-331 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-220 --eval_type train_aug_subsample
# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_aug3_unmemorized_eq_1_total2646_epochs5/checkpoint-110 --eval_type train_aug_subsample


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
