export OMP_NUM_THREADS=1

wandb online
cd /home/chenjian/multi-modal_ECG/merl/MERL/pretrain
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=101 --rdzv_endpoint=localhost:2902 main.py
