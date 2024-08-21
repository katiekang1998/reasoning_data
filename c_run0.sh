
torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-5 --epochs 3 --lora True

RUN_NAME=gsm8k_orig_6epochs_full_lr5e-05_bs128_lora

for CKPT in 58 116 174
do
    CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
    CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
    CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
    CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
    wait 
done



torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 5e-5 --epochs 3 --train_type half

RUN_NAME=gsm8k_orig_6epochs_half_lr5e-05_bs128

for CKPT in 58 116 174
do
    CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
    CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
    CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
    CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
    wait 
done




torchrun --nproc_per_node=4 --master_port=1234 gsm8k_train.py --learning_rate 2e-4 --epochs 3 --lora True

RUN_NAME=gsm8k_orig_6epochs_full_lr0.0002_bs128_lora

for CKPT in 58 116 174
do
    CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
    CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
    CUDA_VISIBLE_DEVICES=2 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
    CUDA_VISIBLE_DEVICES=3 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 1 --temp 0 &
    wait 
done

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
#     CUDA_VISIBLE_DEVICES=1 python gsm8k_eval.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 50 &
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
