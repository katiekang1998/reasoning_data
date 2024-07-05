
NUM_TRAIN_POINTS=20000
for HARD_RATIO in 1. 0.75 0.5 0.25 0.
do
    EASY_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)
    MEDIUM_RATIO=$(echo "(1 - $HARD_RATIO) / 2" | bc -l)

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=1234 math_aug_train.py --num_train_points ${NUM_TRAIN_POINTS} --easy_ratio ${EASY_RATIO} --medium_ratio ${MEDIUM_RATIO} --hard_ratio ${HARD_RATIO}
    
    EASY_RATIO=$(printf "%.2f" $EASY_RATIO)
    MEDIUM_RATIO=$(printf "%.2f" $MEDIUM_RATIO)
    HARD_RATIO=$(printf "%.2f" $HARD_RATIO)

    CKPT_NAME="ckpts/math_aug_easy${EASY_RATIO}_medium${MEDIUM_RATIO}_hard${HARD_RATIO}_total${NUM_TRAIN_POINTS}"
    CUDA_VISIBLE_DEVICES=4,5 python math_eval.py --eval_type train_aug_subsample --ckpt_dir ${CKPT_NAME} &
    P1=$!
    CUDA_VISIBLE_DEVICES=6,7 python math_eval.py --eval_type test --ckpt_dir ${CKPT_NAME} &
    P2=$!
    wait $P1 $P2
    mkdir -p "/nfs/kun2/users/katiekang/reasoning_data_ckpts/${CKPT_NAME}"
    for file in $(find ${CKPT_NAME} -type f -name "*.safetensors"); do
        sudo rsync --progress "$file" /nfs/kun2/users/katiekang/reasoning_data_ckpts/${CKPT_NAME}/
        sudo rm "$file"
    done
    for file in $(find ${CKPT_NAME} -type f -name "*.json"); do
        sudo rsync --progress "$file" /nfs/kun2/users/katiekang/reasoning_data_ckpts/${CKPT_NAME}/
        sudo rm "$file"
    done
    for file in $(find ${CKPT_NAME} -type f -name "*.bin"); do
        sudo rsync --progress "$file" /nfs/kun2/users/katiekang/reasoning_data_ckpts/${CKPT_NAME}/
        sudo rm "$file"
    done
done

# 5000 10000 20000 40000