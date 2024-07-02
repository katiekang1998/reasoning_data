torchrun --nproc_per_node=4 --master_port=1235 math_all_train.py
torchrun --nproc_per_node=4 --master_port=1235 math_train.py --num_epochs 15
torchrun --nproc_per_node=4 --master_port=1235 math_train.py --num_epochs 10
torchrun --nproc_per_node=4 --master_port=1235 math_train.py --num_epochs 5