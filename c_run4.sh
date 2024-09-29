CUDA_VISIBLE_DEVICES=0 python math_eval_perplexity.py --ckpt_dir math_orig_2epochs_test_lr2e-05_bs128 --ckpt 0 --eval_type easy &
CUDA_VISIBLE_DEVICES=1 python math_eval_perplexity3.py --ckpt_dir math_orig_2epochs_test_lr2e-05_bs128 --ckpt 0 --eval_type easy &
wait

# CUDA_VISIBLE_DEVICES=0 python math_eval.py --ckpt_dir ckpts/math_amrith_easy_deepseek_3epochs_7copies_lr2e-05_bs128/checkpoint-657 --eval_type test_easy --num_samples 5 --seed 4 &
# CUDA_VISIBLE_DEVICES=1 python math_eval.py --ckpt_dir ckpts/math_amrith_easy_deepseek_3epochs_13copies_lr2e-05_bs128/checkpoint-1149 --eval_type test_easy --num_samples 5 --seed 4 &
# CUDA_VISIBLE_DEVICES=2 python math_eval.py --ckpt_dir ckpts/math_amrith_easy_deepseek_3epochs_19copies_lr2e-05_bs128/checkpoint-1641 --eval_type test_easy --num_samples 5 --seed 4 &
# wait


# CKPT_NAME="math_aug_easy0.17_medium0.33_hard0.50_total40000"

# find /data/katie_kang/reasoning_data/ckpts/${CKPT_NAME} -maxdepth 2 -type f -name "*.safetensors" -delete