import yaml as yaml
import sys
sys.path.append("../finetune/")
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def get_class_emd(model, class_name, device='cuda'):
    model.eval()
    with torch.no_grad(): # to(device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")) 
        zeroshot_weights = []
        # compute embedding through model for each class
        for texts in tqdm(class_name):
            texts = texts.lower()
            texts = [texts] # convert to list
            texts = model._tokenize(texts) # tokenize
            # class_embeddings, _ = model.get_text_emb(texts.input_ids.to(device=device)
            #                                                 , texts.attention_mask.to(device=device)
            class_embeddings, = model.get_text_emb(texts.input_ids.to(device=device)
                                                            , texts.attention_mask.to(device=device)
                                                            ) # embed with text encoder
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
        
        report_tokenize_output = model._tokenize(reports)
        input_ids = report_tokenize_output.input_ids.to(
            device).contiguous()
        attention_mask = report_tokenize_output.attention_mask.to(
            device).contiguous()
        class_embeddings = model.get_text_emb(input_ids.to(device=device)
                                                        , attention_mask.to(device=device)
                                                        ) # embed with text encoder
        class_embeddings = model.proj_t(class_embeddings) # embed with text encoder

        # normalize class_embeddings
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

        logits = class_embeddings @ class_weight.to(device)
        logits = torch.squeeze(logits, 0) # (N, num_classes)
        norm_logits = (logits - logits.mean()) / (logits.std())
        logits = torch.sigmoid(norm_logits) 
            
        y_pred.append(logits.cpu().data.numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    labels = np.array(y_pred)
    labels = np.argmax(labels, axis=1)
    return labels, y_pred


def get_metrics(label, predict):
    f1 = f1_score(label, predict, 'macro')
    pre = precision_score(label, predict)
    rec = recall_score(label, predict)
    return f1, pre, rec