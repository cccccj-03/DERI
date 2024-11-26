import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import yaml
from models.model import ECGCLIP
from model_bert import ERbert
from torch.utils.data.dataloader import DataLoader
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
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

sys.path.append("../utils")
import utils_dataset

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
# ckpt_path = '/home/chenjian/multi-modal_ECG/merl/MERL/chec/resnet_7_mix_sep_bestZeroShotAll_ckpt.pth'
# ckpt = torch.load(ckpt_path, map_location='cpu')
# ckpt_path = '/data/lizixuan/cj/ECG_MM/res18_best_ckpt.pth'
# ckpt = torch.load(ckpt_path, map_location='cpu')
# config = yaml.load(open("/home/lizixuan/chenjian/ecg_mm/MERL/report_generate/config.yaml", "r"), Loader=yaml.FullLoader)
# encoder = ECGCLIP(config['network'])
ckpt_path = '/home/chenjian/multi-modal_ECG/merl/MERL/zeroshot/78.72/resnet_mix_sep_bestZeroShotAll_ckpt.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
config = yaml.load(open("/home/chenjian/multi-modal_ECG/merl/MERL/pretrain/config.yaml", "r"), Loader=yaml.FullLoader)
encoder = ECGCLIP(config['network'])
encoder.load_state_dict(ckpt, strict=False)

# decoder_path = '/data/lizixuan/cj/distilgpt2'
# model = ERGPT2(encoder=encoder, decoder_path=decoder_path)

decoder_path = '/home/chenjian/multi-modal_ECG/MedCPT-Query-Encoder'

model = ERbert(encoder=encoder, decoder_path=decoder_path)


optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-8)


scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=1e-5, last_epoch=-1)
# scaler = GradScaler()
model = model.to(device_id)

# prompt_type = 'CKEPE'
# prompt_dict = '/home/lizixuan/chenjian/ecg_mm/MERL/report_generate/CKEPE_prompt.json'
# with open(prompt_dict, 'r') as f:
#     prompt_dict = yaml.load(f, Loader=yaml.FullLoader)
# target_class = [class_name for class_name in prompt_dict.values()]
# class_weight = get_class_emd(model=model, class_name=target_class, device=rank)

model = DDP(model, device_ids=[device_id], find_unused_parameters=True)


## Train parameter
batch_size = 64
val_batch_size = 16
checkpoint_interval = 50
max_epochs = 20
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




# automatically resume from checkpoint if it exists
print('#########################################')

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

val_loss = 9999999
for epoch_counter in tqdm(range(start_epoch, max_epochs)):

    epoch_loss = 0
    model.train()
    for data in tqdm(train_loader):
        model.train()
        report = data['raw_text']#.to(rank)

        # get ecg
        ecg = data['ecg'].to(torch.float32).contiguous().to(rank)
        # decoder_input_ids, decoder_attention_mask = prepare_decoder_input(report, tokenizer=tokenizer)
        # print(ecg.shape)
        optimizer.zero_grad()
        with autocast():

            decoder_input_ids, decoder_attention_mask = model.module.prepare_decoder_input(report)
            output = model(ecg,
                    decoder_input_ids.to(ecg.device), mode='train')
            loss = model.module.loss(
                    output, decoder_input_ids.to(ecg.device), decoder_attention_mask.to(ecg.device))

            world_size = torch_dist.get_world_size()


            if rank == 0:
                print(f'loss is {loss.item():.4f}')

            epoch_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()


    ## Eval
    
    val_epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(val_loader):
            model.eval()
            report = data['raw_text']#.to(rank)
            # get ecg
            ecg = data['ecg'].to(torch.float32).contiguous().to(rank)
            # decoder_input_ids, decoder_attention_mask, label_ids = prepare_decoder_input(report, tokenizer=tokenizer)
            # print(ecg.shape)
            with torch.no_grad():
                decoder_input_ids, decoder_attention_mask = model.module.prepare_decoder_input(report)
                output = model(ecg,
                            decoder_input_ids.to(ecg.device), mode='train')
                loss = model.module.loss(
                        output, decoder_input_ids.to(ecg.device), decoder_attention_mask.to(ecg.device))

            val_epoch_loss += loss.item()

    print('val epoch loss:', val_epoch_loss)
    # 打印平均结果

    if val_epoch_loss <= val_loss:
        print(f'val loss reduced from {val_loss:.4f} to {val_epoch_loss:.4f}:')
        val_loss = val_epoch_loss 
        if epoch_counter == 0:
            torch.save({
                'epoch': epoch_counter,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                f'/data/chenjian/ECG_MM/report/ckpt/MedCPT_0_ckpt.pth'     
                )
        else:
            torch.save({
                'epoch': epoch_counter,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                f'/data/chenjian/ECG_MM/report/ckpt/MedCPT_ckpt.pth'     
                )
        best_model = model


# labels_all = []
# reports_all = []
# outputs_all = []
# labels_pred_all = []
# with torch.no_grad():
#     for data in tqdm(val_loader):
#         model.eval()
#         report = data['raw_text']#.to(rank)
#         # get ecg
#         ecg = data['ecg'].to(torch.float32).contiguous().to(rank)
#         with torch.no_grad():
#             output_r = model.module.generate(ecg)
#             reports_all.append(report)
#             outputs_all.append(output_r)
#             label, logits = get_ground_truth(model=model.module, reports=report, class_weight=class_weight, device=rank)
#             label_pred, logits_pred = get_ground_truth(model=model.module, reports=output_r , class_weight=class_weight, device=rank)
#             labels_all.append(label)
#             labels_pred_all.append(label_pred)

# reports_all = [i for item in reports_all for i in item]
# outputs_all = [i for item in outputs_all for i in item]
# labels_a = np.hstack(labels_all)
# labels_p = np.hstack(labels_pred_all)

# average_bleu1, average_bleu2, average_bleu3, average_bleu4, average_rougeL = report_matrix(reports_all, outputs_all)
# f1 = f1_score(labels_a, labels_p, average='macro')
# pre = precision_score(labels_a, labels_p, average='macro')
# rec = recall_score(labels_a, labels_p, average='macro')    
            

# print(f"Average BLEU-1 Score: {average_bleu1:.4f}")
# print(f"Average BLEU-2 Score: {average_bleu2:.4f}")
# print(f"Average BLEU-3 Score: {average_bleu3:.4f}")
# print(f"Average BLEU-4 Score: {average_bleu4:.4f}")
# print(f"Average ROUGE-L F1 Score: {average_rougeL:.4f}")
# print(f"CE F1 Score: {f1:.4f}")
# print(f"CE Precision Score: {pre:.4f}")
# print(f"CE Recall Score: {rec:.4f}")

# metrics = {'BLEU-1':average_bleu1,
#         'BLEU-2':average_bleu2,
#         'BLEU-3':average_bleu3,
#         'BLEU-4':average_bleu4,
#         'ROUGE-L F1 Score':average_rougeL,
#         'CE F1 Score': f1,
#         'CE Precision Score': pre,
#         'CE F1 Score': rec,
#         'report':reports_all,
#         'generated report': outputs_all
#         }

# torch.save({
#     'metrics': metrics},
#     f'/data/lizixuan/cj/ECG_MM/checkpoints/SciBert_metrics.pth'     
#     )