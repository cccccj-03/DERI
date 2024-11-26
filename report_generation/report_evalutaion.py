import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import yaml
from models.model import ECGCLIP
from model_bert import ERbert
# from model_clinicalbert import ERclinicalbert
# from model_pubmedbert import ERPubMedbert
from torch.utils.data.dataloader import DataLoader
from loss import get_class_emd, get_ground_truth, report_matrix
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast
from torch.utils.data.distributed import DistributedSampler
# from torch import distributed as torch_dist

from tqdm import tqdm
import pandas as pd
import numpy as np
import random
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from sklearn.metrics import f1_score, precision_score, recall_score
import datetime
sys.path.append("../utils")
import utils_dataset

torch.manual_seed(42)
random.seed(0)
np.random.seed(0)
# torch.backends.cudnn.benchmark = True

# os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
# dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=7200))

torch.cuda.empty_cache()
# rank = dist.get_rank()

#print(f"Start running basic DDP example on rank {rank}.")
device = 'cuda' #rank % torch.cuda.device_count()

## load data 
data_path = '/data/chenjian/ECG_MM/pretrain_data/'
dataset_name = 'mimic'
dataset = utils_dataset.ECG_TEXT_Dsataset(
        data_path=data_path, dataset_name=dataset_name)
# train_dataset = dataset.get_dataset(train_test='train')
val_dataset = dataset.get_dataset(train_test='val')


## load model
# ckpt_path = '/home/chenjian/multi-modal_ECG/merl/MERL/chec/resnet_7_mix_sep_bestZeroShotAll_ckpt.pth'
# ckpt = torch.load(ckpt_path, map_location='cpu')
# ckpt_path = '/data/lizixuan/cj/ECG_MM/res18_best_ckpt.pth'
# ckpt = torch.load(ckpt_path, map_location='cpu')
# config = yaml.load(open("/home/lizixuan/chenjian/ecg_mm/MERL/report_generate/config.yaml", "r"), Loader=yaml.FullLoader)
# encoder = ECGCLIP(config['network'])

# ckpt_path = '/home/chenjian/multi-modal_ECG/merl/MERL/chec/res18_best_ckpt.pth'
# ckpt = torch.load(ckpt_path, map_location='cpu')
config = yaml.load(open("/home/chenjian/multi-modal_ECG/merl/MERL/finetune/config.yaml", "r"), Loader=yaml.FullLoader)
encoder = ECGCLIP(config['network'])

# encoder.load_state_dict(ckpt, strict=False)

# decoder_path = '/data/lizixuan/cj/distilgpt2'
# model = ERGPT2(encoder=encoder, decoder_path=decoder_path)

decoder_path = '/home/chenjian/multi-modal_ECG/MedCPT-Query-Encoder'

model = ERbert(encoder=encoder, decoder_path=decoder_path)


optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-5)


# scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=1e-3, last_epoch=-1)
# scaler = GradScaler()
ckpt_model = torch.load(r'/data/chenjian/ECG_MM/report/ckpt/MedCPT_ckpt.pth', map_location='cpu')
ckpt_model = ckpt_model['model_state_dict']
model.load_state_dict(ckpt_model)

model = model.to(device)


prompt_type = 'CKEPE'
prompt_dict = '/home/chenjian/multi-modal_ECG/merl/MERL/zeroshot/CKEPE_prompt.json'
with open(prompt_dict, 'r') as f:
    prompt_dict = yaml.load(f, Loader=yaml.FullLoader)
target_class = [class_name for class_name in prompt_dict.values()]
class_weight = get_class_emd(model=model, class_name=target_class, device=device)

#model = DDP(model, device_ids=[device_id], find_unused_parameters=True)


## Train parameter
val_batch_size = 2
checkpoint_interval = 50
max_epochs = 20
num_workers = 0

## data loader 
val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                        num_workers=num_workers,
                        drop_last=True, shuffle=False, pin_memory=True)



# automatically resume from checkpoint if it exists
print('#########################################')


### generate ground truth
labels_all = []
reports_all = []
outputs_all = []
labels_pred_all = []
n = 0
with torch.no_grad():
    for data in tqdm(val_loader):
        with torch.no_grad():
            if n >= 6:
                continue
            n += 1
            model.eval()
            report = data['raw_text']#.to(rank)
            # get ecg
            ecg = data['ecg'].to(torch.float32).contiguous().to(device)
        with torch.no_grad():
            output_r = model.generate(ecg)
            reports_all.append(report)
            outputs_all.append(output_r)
            label, logits = get_ground_truth(model=model, reports=report, class_weight=class_weight, device=device)
            label_pred, logits_pred = get_ground_truth(model=model, reports=output_r , class_weight=class_weight, device=device)
            labels_all.append(label)
            labels_pred_all.append(label_pred)

reports_all = [i for item in reports_all for i in item]
outputs_all = [i for item in outputs_all for i in item]
labels_a = np.hstack(labels_all)
labels_p = np.hstack(labels_pred_all)

average_bleu1, average_bleu2, average_bleu3, average_bleu4, average_rougeL = report_matrix(reports_all, outputs_all)
f1 = f1_score(labels_a, labels_p, average='macro')
pre = precision_score(labels_a, labels_p, average='macro')
rec = recall_score(labels_a, labels_p, average='macro')    
            

# average_bleu1 = torch.tensor(average_bleu1).to(device)
# average_bleu2 = torch.tensor(average_bleu2).to(device)
# average_bleu3 = torch.tensor(average_bleu3).to(device)
# average_bleu4 = torch.tensor(average_bleu4).to(device)
# average_rougeL = torch.tensor(average_rougeL).to(device)
# f1 = torch.tensor(f1).to(device)
# pre = torch.tensor(pre).to(device)
# rec = torch.tensor(rec).to(device)

# 定义一个all_reduce函数来聚合并求平均
# def average_metric(metric):
#     dist.all_reduce(metric, op=dist.ReduceOp.SUM)  # 将所有进程的指标求和
#     metric /= dist.get_world_size()  # 除以进程数，得到平均值
#     return metric

# 聚合所有指标
# average_bleu1 = average_bleu1.item()
# average_bleu2 = average_metric(average_bleu2).item()
# average_bleu3 = average_metric(average_bleu3).item()
# average_bleu4 = average_metric(average_bleu4).item()
# average_rougeL = average_metric(average_rougeL).item()
# f1 = average_metric(f1).item()
# pre = average_metric(pre).item()
# rec = average_metric(rec).item()


print(f"Average BLEU-1 Score: {average_bleu1:.4f}")
print(f"Average BLEU-2 Score: {average_bleu2:.4f}")
print(f"Average BLEU-3 Score: {average_bleu3:.4f}")
print(f"Average BLEU-4 Score: {average_bleu4:.4f}")
print(f"Average ROUGE-L F1 Score: {average_rougeL:.4f}")
print(f"CE F1 Score: {f1:.4f}")
print(f"CE Precision Score: {pre:.4f}")
print(f"CE Recall Score: {rec:.4f}")

metrics = {'BLEU-1':average_bleu1,
        'BLEU-2':average_bleu2,
        'BLEU-3':average_bleu3,
        'BLEU-4':average_bleu4,
        'ROUGE-L F1 Score':average_rougeL,
        'CE F1 Score': f1,
        'CE Precision Score': pre,
        'CE Rec Score': rec,
        'report':reports_all,
        'generated report': outputs_all
        }

torch.save({
    'metrics': metrics},
    f'/data/chenjian/ECG_MM/report/ckpt/MedCPT_metrics.pth'     
    )