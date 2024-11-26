task_name=$1
backbone=$2
pretrain_path=$3
ckpt_dir="/home/chenjian/multi-modal_ECG/merl/MERL/finetune/ckpt/chapman/$task_name"

python /home/chenjian/multi-modal_ECG/merl/MERL/finetune/main_single.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset chapman \
    --pretrain_path $pretrain_path \
    --ratio 1 \
    --learning-rate 0.01 \
    --backbone $backbone \
    --epochs 100 \
    --name $task_name

# python /home/chenjian/multi-modal_ECG/merl/MERL/finetune/main_single.py \
#     --checkpoint-dir $ckpt_dir \
#     --batch-size 16 \
#     --dataset chapman \
#     --pretrain_path $pretrain_path \
#     --ratio 10 \
#     --learning-rate 0.01 \
#     --backbone $backbone \
#     --epochs 100 \
#     --name $task_name

# python /home/chenjian/multi-modal_ECG/merl/MERL/finetune/main_single.py \
#     --checkpoint-dir $ckpt_dir \
#     --batch-size 16 \
#     --dataset chapman \
#     --pretrain_path $pretrain_path \
#     --ratio 100 \
#     --learning-rate 0.01 \
#     --backbone $backbone \
#     --epochs 100 \
#     --name $task_name