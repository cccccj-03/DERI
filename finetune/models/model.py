from cgi import test
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from models.resnet1d import ResNet18, ResNet34, ResNet50, ResNet101
from models.vit1d import vit_base, vit_small, vit_tiny, vit_middle
from models.block import Transformer

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, spacial_dim + 1, embed_dim) / embed_dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)        
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        
        
        self.positional_embedding = nn.Parameter(torch.randn(1, spacial_dim + 2, embed_dim) / embed_dim)
        self.sep_embedding = nn.Parameter(torch.randn(embed_dim))
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # convert X shape (B, C, L) to (B, L, C)
        
        x = x + self.positional_embedding[:,1:-1,:]
        sep_embedding = self.sep_embedding[None,  None, :]
        left_sep = sep_embedding.expand(x.shape[0], -1, -1) + self.positional_embedding[:,  :1, :]
        right_sep = sep_embedding.expand(x.shape[0], -1, -1) + self.positional_embedding[:, -1:, :]
        x = torch.cat([left_sep, x, right_sep], dim=1)
        
        # self.cls_tokens = self.cls_token + self.positional_embedding[:, :1, :]
        # self.cls_tokens = self.cls_tokens.expand(x.shape[0], -1, -1) 
        # x = torch.cat((self.cls_tokens, x), dim=1)
        # x = x + self.positional_embedding[:, :, :].to(x.dtype)  # (L+1)NC
        x, att_map = self.mhsa(x, x, x, average_attn_weights=True)
        # x = self.c_proj(x)
        return x[:,1:-1,]#, att_map[:, :, 1:]
    
class ECGCLIP(torch.nn.Module):
    def __init__(self, network_config):
        super(ECGCLIP, self).__init__()
        
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']

        # ecg signal encoder
        self.ecg_model = network_config['ecg_model']
        self.num_leads = network_config['num_leads']
        self.mask_ratio = 0.1
        if 'resnet' in self.ecg_model:
            if self.ecg_model == 'resnet18':
                model = ResNet18()
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1, stride=3)
                self.att_pool_head = AttentionPool2d(spacial_dim=105,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            elif self.ecg_model == 'resnet34':
                model = ResNet34()
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1, stride=3)
                self.att_pool_head = AttentionPool2d(spacial_dim=105,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            elif self.ecg_model == 'resnet50':
                model = ResNet50()
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1, stride=3)
                self.att_pool_head = AttentionPool2d(spacial_dim=105,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            elif self.ecg_model == 'resnet101':
                model = ResNet101()
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1, stride=3)
                self.att_pool_head = AttentionPool2d(spacial_dim=105,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)

            self.linear1 = AttentionPool2d(spacial_dim=int(105*(1-self.mask_ratio)),
                                                    embed_dim=self.proj_out, 
                                                    num_heads=2, 
                                                    output_dim=self.proj_out)
            #nn.Linear(self.proj_out, self.proj_out, bias=False)
            self.linear2 = AttentionPool2d(spacial_dim=int(105*(1-self.mask_ratio)),
                                                    embed_dim=self.proj_out, 
                                                    num_heads=2, 
                                                    output_dim=self.proj_out)
            #nn.Linear(self.proj_out, self.proj_out, bias=False)

            self.decode_t = Transformer(num_patches=105, width=self.proj_out, out_dim=768, mlp_dim=256, depth=2)
            self.decode_e = Transformer(num_patches=256, width=self.proj_out, out_dim=self.proj_out, mlp_dim=256, depth=2)


        if 'vit' in self.ecg_model:
            if self.ecg_model == 'vit_tiny':
                model = vit_tiny(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_small':
                model = vit_small(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_middle':
                model = vit_middle(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_base':
                model = vit_base(num_leads=self.num_leads)
            self.proj_e_input = model.width    
            self.proj_e = nn.Sequential(
                nn.Linear(self.proj_e_input, self.proj_hidden),
                nn.BatchNorm1d(self.proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.proj_hidden, self.proj_out),
                nn.BatchNorm1d(self.proj_out),
            )
            self.linear1 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)


        self.ecg_encoder = model
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        # text encoder
        url = network_config['text_model']
        self.lm_model = AutoModel.from_pretrained(
            url, trust_remote_code=True, revision='main')
        self.tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True, revision='main')
        
        # text projector
        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )

        self.head = nn.Identity()

    def forward_feature(self, ecg):

        if 'resnet' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)
            ecg_emb = self.downconv(ecg_emb)
            proj_ecg_emb = self.att_pool_head(ecg_emb)
            # proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)

        if 'vit' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)
            proj_ecg_emb = self.proj_e(ecg_emb)

        #sep_embedding = self.decode_e.sep_embedding[None,  None, :].expand(proj_ecg_emb.shape[0], -1, -1)
        #dec_text_emb = torch.concat([proj_ecg_emb, sep_embedding], dim=1)
        dec_text_emb = self.decode_t(proj_ecg_emb)
        proj_ecg_emb = torch.mean(proj_ecg_emb, dim=1)
        dec_text_emb = torch.mean(dec_text_emb, dim=1)
        mix_ecg_emb = proj_ecg_emb + self.proj_t(dec_text_emb) 
        return mix_ecg_emb
    
    def forward(self, ecg):
        x = self.forward_feature(ecg)
        x = self.head(x)
        return x
    
    def reset_head(self, num_classes=1):
        del self.head
        self.head = nn.Linear(self.proj_out, num_classes)