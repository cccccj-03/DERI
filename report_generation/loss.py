import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml as yaml
import sys
sys.path.append("../finetune/")
from tqdm import tqdm
import numpy as np

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def infonce_loss(out_1, out_2, softmax_temperature):
    batch_size = out_1.size(0)
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    sim = out_2.detach() @ out_2.detach().t()
    lambda_ = 1.
    targets = lambda_ * \
        torch.eye(batch_size).type_as(sim) + (1 - lambda_) * sim

    logits = out_1 @ out_2.t()
    loss0 = F.cross_entropy(logits / softmax_temperature, targets)
    loss1 = F.cross_entropy(logits.t() / softmax_temperature, targets)
    cont_loss = (loss0 + loss1) / 2.

    return cont_loss


def ce_loss(y_pred, ids, pad_token_id=None):
    #loss_fn = nn.CrossEntropyLoss()
    if pad_token_id != None:
        loss = F.cross_entropy(y_pred.permute([0, 2, 1]).contiguous(), ids, ignore_index=pad_token_id)
    else:
        loss = F.cross_entropy(y_pred.permute([0, 2, 1]).contiguous(), ids)

    return loss




def get_class_emd(model, class_name, device='cuda'):
    model.eval()
    with torch.no_grad(): # to(device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")) 
        zeroshot_weights = []
        # compute embedding through model for each class
        for texts in tqdm(class_name):
            texts = texts.lower()
            texts = [texts] # convert to list
            texts = model.encoder_tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=256,
                                                            padding='max_length',
                                                            return_tensors='pt') # tokenize
            class_embeddings = model.text_encoder(input_ids=texts['input_ids'].to(device),
                                 attention_mask=texts['attention_mask'].to(device)).pooler_output # embed with text encoder
            class_embeddings = model.proj_t(class_embeddings) # embed with text encoder

            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates 
            class_embedding = class_embeddings.mean(dim=0) 
            # norm over new averaged templates
            class_embedding /= class_embedding.norm() 
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights



def get_ground_truth(model, reports, class_weight, device='cuda'):
    model.eval()
    y_pred = []
    with torch.no_grad():
        
        report_tokenize_output =model.encoder_tokenizer.batch_encode_plus(batch_text_or_text_pairs=reports,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=256,
                                                            padding='max_length',
                                                            return_tensors='pt')
        input_ids = report_tokenize_output.input_ids.to(
            device).contiguous()
        attention_mask = report_tokenize_output.attention_mask.to(
            device).contiguous()
        class_embeddings = model.text_encoder(input_ids,
                                 attention_mask).pooler_output
                                                         # embed with text encoder
        class_embeddings = model.proj_t(class_embeddings) # embed with text encoder

        # normalize class_embeddings
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        logits = class_embeddings @ class_weight.to(device)
        logits = torch.squeeze(logits, 0) # (N, num_classes)
        # norm_logits = (logits - logits.mean()) / (logits.std())
        # logits = torch.sigmoid(norm_logits) 
            
        y_pred.append(logits.cpu().data.numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    labels = np.array(y_pred)
    labels = np.argmax(labels, axis=1)
    return labels, y_pred




def report_matrix(reports_all, outputs_all):
    # 初始化 BLEU 和 ROUGE scorer
    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # 存储 BLEU 和 ROUGE 分数
    bleu_scores = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []}
    rougeL_scores = []

    # 逐一计算每个样本的 BLEU 和 ROUGE 分数
    for ref, gen in zip(reports_all, outputs_all):
        # 计算 BLEU-1 分数
        bleu1 = sentence_bleu([ref.split()], gen.split(), weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu_scores['bleu1'].append(bleu1)
        
        # 计算 BLEU-2 分数
        bleu2 = sentence_bleu([ref.split()], gen.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        bleu_scores['bleu2'].append(bleu2)
        
        # 计算 BLEU-3 分数
        bleu3 = sentence_bleu([ref.split()], gen.split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
        bleu_scores['bleu3'].append(bleu3)
        
        # 计算 BLEU-4 分数
        bleu4 = sentence_bleu([ref.split()], gen.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        bleu_scores['bleu4'].append(bleu4)
        
        # 计算 ROUGE-L 分数
        rougeL = scorer.score(ref, gen)['rougeL'].fmeasure
        rougeL_scores.append(rougeL)

    # 计算平均值
    average_bleu1 = sum(bleu_scores['bleu1']) / len(bleu_scores['bleu1'])
    average_bleu2 = sum(bleu_scores['bleu2']) / len(bleu_scores['bleu2'])
    average_bleu3 = sum(bleu_scores['bleu3']) / len(bleu_scores['bleu3'])
    average_bleu4 = sum(bleu_scores['bleu4']) / len(bleu_scores['bleu4'])
    average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return average_bleu1, average_bleu2, average_bleu3, average_bleu4, average_rougeL