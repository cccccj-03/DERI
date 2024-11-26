import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (GPT2Config, GPT2TokenizerFast,
                          GPT2LMHeadModel, PretrainedConfig, EncoderDecoderModel)
from transformers.modeling_outputs import BaseModelOutput
from metrics import get_class_emd, get_ground_truth
from models.model import ECGCLIP
import sys
sys.path.append("../utils")
import utils_dataset
import yaml
# from models.resnet1d import ResNet18, ResNet34, ResNet50, ResNet101

from model_gpt2 import ERGPT2
from torch.utils.data.dataloader import DataLoader
from loss import infonce_loss, ce_loss
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as torch_dist
import torch.distributed as dist

from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import wandb
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"



torch.manual_seed(42)
random.seed(0)
np.random.seed(0)


dist.init_process_group("nccl")
torch.cuda.empty_cache()
rank = dist.get_rank()

print(f"Start running basic DDP example on rank {rank}.")
device_id = rank % torch.cuda.device_count()

## load data 
data_path = '/data/chenjian/ECG_MM/pretrain_data/'
dataset_name = 'mimic'
dataset = utils_dataset.ECG_TEXT_Dsataset(
        data_path=data_path, dataset_name=dataset_name)
train_dataset = dataset.get_dataset(train_test='train')
val_dataset = dataset.get_dataset(train_test='val')


## load model
ckpt_path = '/home/chenjian/multi-modal_ECG/merl/MERL/zeroshot/78.72/resnet_mix_sep_bestZeroShotAll_ckpt.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
# ckpt_path = '/home/chenjian/multi-modal_ECG/merl/MERL/chec/res18_best_ckpt.pth'
# ckpt = torch.load(ckpt_path, map_location='cpu')
# config = yaml.load(open("/home/chenjian/multi-modal_ECG/merl/MERL/finetune/config.yaml", "r"), Loader=yaml.FullLoader)
# encoder = ECGCLIP(config['network'])
config = yaml.load(open("/home/chenjian/multi-modal_ECG/merl/MERL/finetune/config.yaml", "r"), Loader=yaml.FullLoader)
encoder = ECGCLIP(config['network'])
encoder.load_state_dict(ckpt, strict=True)

decoder_path = '/home/chenjian/multi-modal_ECG/distilgpt2'

model = ERGPT2(encoder=encoder, decoder_path=decoder_path)
# for param in model.encoder.parameters():
    # param.requires_grad = False
model.encoder.head.weight.requires_grad = True
model.encoder.head.bias.requires_grad = True   

optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-8)


scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=1e-3, last_epoch=-1)
scaler = GradScaler()
model = model.to(device_id)
model = DDP(model, device_ids=[device_id], find_unused_parameters=True)


## Train parameter
batch_size = 16
val_batch_size = 16
checkpoint_interval = 50
max_epochs = 1
num_workers = 8

## data loader 
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True, shuffle=False,
                                  sampler=DistributedSampler(train_dataset))
        
val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                        num_workers=num_workers,
                        drop_last=True, shuffle=False,
                        sampler=DistributedSampler(val_dataset))


# model_checkpoints_folder = '/home/chenjian/multi-modal_ECG/merl/MERL/report_generate/ckpt'

# if not os.path.exists(model_checkpoints_folder):
#     print('create directory "{}" for save checkpoint!'.format(
#         model_checkpoints_folder))
#     print('---------------------------')
#     os.makedirs(model_checkpoints_folder)
# else:
#     print('directory "{}" existing for save checkpoint!'.format(
#         model_checkpoints_folder))

# automatically resume from checkpoint if it exists
# print('#########################################')
# print('Be patient..., checking checkpoint now...')
# if os.path.exists(model_checkpoints_folder +'_checkpoint.pth'):
#     ckpt = torch.load(model_checkpoints_folder+'_checkpoint.pth',
#                         map_location='cpu')
#     start_epoch = ckpt['epoch']
#     model.load_state_dict(ckpt['model_state_dict'])
#     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#     print('continue training successful!')
# else:
start_epoch = 0
print('Start training from 0 epoch')


scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5000,
            T_mult=1,
            eta_min=1e-8,
        )
niter = 1

skip_scheduler = False
scaler = GradScaler()


### generate ground truth
# prompt_type = 'CKEPE'
# prompt_dict = '/home/chenjian/multi-modal_ECG/merl/MERL/zeroshot/CKEPE_prompt.json'
# with open(prompt_dict, 'r') as f:
#     prompt_dict = yaml.load(f, Loader=yaml.FullLoader)
# target_class = [class_name for class_name in prompt_dict.values()]
# # class_weight = get_class_emd(model=model.module.encoder, class_name=target_class, device=rank)

val_loss = 9999999
for epoch_counter in tqdm(range(start_epoch, max_epochs+1)):
    epoch_loss = 0
    model.train()
    for data in tqdm(train_loader):
        model.train()
        report = data['raw_text']#.to(rank)

        # get ecg
        ecg = data['ecg'].to(torch.float32).contiguous().to(rank)
        # decoder_input_ids, decoder_attention_mask, label_ids = prepare_decoder_input(report, tokenizer=tokenizer)
        # print(ecg.shape)
        optimizer.zero_grad()
        with autocast():
            logits, decoder_input_ids, decoder_attention_mask, label_ids = model(ecg, report)
            # decoder_attention_mask = decoder_attention_mask.to(rank)
            # decoder_input_ids = decoder_input_ids.to(rank)
            # label_ids = label_ids.to(rank)
            world_size = torch_dist.get_world_size()

            with torch.no_grad():
                agg_logits = [torch.zeros_like(logits) for _ in range(world_size)]
                agg_decoder_input_ids = [torch.zeros_like(decoder_input_ids) for _ in range(world_size)]
                agg_decoder_attention_mask = [torch.zeros_like(decoder_attention_mask) for _ in range(world_size)]
                agg_label_ids = [torch.zeros_like(label_ids) for _ in range(world_size)]
            
                dist.all_gather(agg_logits, logits)
                dist.all_gather(agg_decoder_input_ids, decoder_input_ids)
                dist.all_gather(agg_decoder_attention_mask, decoder_attention_mask)
                dist.all_gather(agg_label_ids, label_ids)
            
            agg_logits[rank] = logits
            agg_decoder_input_ids[rank] = decoder_input_ids
            agg_decoder_attention_mask[rank] = decoder_attention_mask
            agg_label_ids[rank] = label_ids

            
            agg_logits = torch.cat(agg_logits, dim=0)
            agg_decoder_input_ids = torch.cat(agg_decoder_input_ids, dim=0)
            agg_decoder_attention_mask = torch.cat(agg_decoder_attention_mask, dim=0)
            agg_label_ids = torch.cat(agg_label_ids, dim=0)

            loss = ce_loss(agg_logits, agg_label_ids, pad_token_id=model.module.tokenizer.pad_token_id)

            if rank == 0:
                print(f'loss is {loss:.4f}')

            epoch_loss += loss#.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    torch.save({
        'epoch': epoch_counter,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        f'/data/chenjian/ECG_MM/report/ckpt/DisGPT2_align_{epoch_counter}_ckpt.pth')    
    ### Eval

    val_epoch_loss = 0
    model.eval()
    for data in tqdm(val_loader):
        model.eval()
        report = data['raw_text']#.to(rank)
        # get ecg
        ecg = data['ecg'].to(torch.float32).contiguous().to(rank)
        # decoder_input_ids, decoder_attention_mask, label_ids = prepare_decoder_input(report, tokenizer=tokenizer)
        # print(ecg.shape)

        logits, decoder_input_ids, decoder_attention_mask, label_ids = model(ecg, report)
        world_size = torch_dist.get_world_size()

        with torch.no_grad():
            agg_logits = [torch.zeros_like(logits) for _ in range(world_size)]
            agg_decoder_input_ids = [torch.zeros_like(decoder_input_ids) for _ in range(world_size)]
            agg_decoder_attention_mask = [torch.zeros_like(decoder_attention_mask) for _ in range(world_size)]
            agg_label_ids = [torch.zeros_like(label_ids) for _ in range(world_size)]
        
            dist.all_gather(agg_logits, logits)
            dist.all_gather(agg_decoder_input_ids, decoder_input_ids)
            dist.all_gather(agg_decoder_attention_mask, decoder_attention_mask)
            dist.all_gather(agg_label_ids, label_ids)
        
        agg_logits[rank] = logits
        agg_decoder_input_ids[rank] = decoder_input_ids
        agg_decoder_attention_mask[rank] = decoder_attention_mask
        agg_label_ids[rank] = label_ids

        agg_logits = torch.cat(agg_logits, dim=0)
        agg_decoder_input_ids = torch.cat(agg_decoder_input_ids, dim=0)
        agg_decoder_attention_mask = torch.cat(agg_decoder_attention_mask, dim=0)
        agg_label_ids = torch.cat(agg_label_ids, dim=0)
        with torch.no_grad():
            loss = ce_loss(agg_logits, agg_label_ids, pad_token_id=model.module.tokenizer.pad_token_id)

        val_epoch_loss += loss#.item()
    
    print('val epoch loss:', val_epoch_loss)

    if val_epoch_loss <= val_loss:
        val_loss = val_epoch_loss
        print(f'val loss reduced from {val_loss:.4f} to {val_epoch_loss:.4f}:')
        torch.save({
            'epoch': epoch_counter,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            f'/data/chenjian/ECG_MM/report/ckpt/DisGPT2_align_{epoch_counter}_ckpt.pth')