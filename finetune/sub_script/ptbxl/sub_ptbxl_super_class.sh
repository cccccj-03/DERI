task_name=$1
backbone=$2
pretrain_path=$3
ckpt_dir="/home/chenjian/multi-modal_ECG/merl/MERL/finetune/ckpt/ptbxl_super_class/$task_name"

python /home/chenjian/multi-modal_ECG/merl/MERL/finetune/main_single.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --ratio 1 \
    --dataset ptbxl_super_class \
    --pretrain_path $pretrain_path \
    --learning-rate 0.01 \
    --backbone $backbone \
    --epochs 100 \
    --name $task_name

# python /home/chenjian/multi-modal_ECG/merl/MERL/finetune/main_single.py \
#     --checkpoint-dir $ckpt_dir \
#     --batch-size 16 \
#     --ratio 10 \
#     --dataset ptbxl_super_class \
#     --pretrain_path $pretrain_path \
#     --learning-rate 0.01 \
#     --backbone $backbone \
#     --epochs 100 \
#     --name $task_name

# python /home/chenjian/multi-modal_ECG/merl/MERL/finetune/main_single.py \
#     --checkpoint-dir $ckpt_dir \
#     --batch-size 16 \
#     --ratio 100 \
#     --dataset ptbxl_super_class \
#     --pretrain_path $pretrain_path \
#     --learning-rate 0.01 \
#     --backbone $backbone \
#     --epochs 100 \
#     --name $task_name