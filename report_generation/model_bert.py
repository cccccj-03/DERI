import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (BertTokenizerFast, BertLMHeadModel, BertConfig, PretrainedConfig, EncoderDecoderModel)
from transformers.modeling_outputs import BaseModelOutput
from models.r2gen.encoder_decoder import EncoderDecoder
from models.r2gen.loss import compute_loss
from models.r2gen.visual_extractor import  VisualExtractor
from argparse import Namespace, ArgumentParser
from copy import deepcopy as c
import numpy as np
from typing import Any

parser = ArgumentParser(
        description="Run downstream task of few shot learning")

    # Model settings (for visual extractor)
parser.add_argument('--visual_extractor', type=str,
                    default='resnet18', help='the visual extractor to be used.')
parser.add_argument('--visual_extractor_pretrained', type=bool,
                    default=True, help='whether to load the pretrained visual extractor')
parser.add_argument('--max_seq_length', type=int,
                    default=256)

parser.add_argument('--d_model', type=int, default=512,
                    help='the dimension of Transformer.')
parser.add_argument('--d_ff', type=int, default=512,
                    help='the dimension of FFN.')
parser.add_argument('--d_vf', type=int, default=512,
                    help='the dimension of the patch features.')
parser.add_argument('--num_heads', type=int, default=8,
                    help='the number of heads in Transformer.')
parser.add_argument('--num_layers', type=int, default=3,
                    help='the number of layers of Transformer.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='the dropout rate of Transformer.')
parser.add_argument('--logit_layers', type=int, default=1,
                    help='the number of the logit layer.')
parser.add_argument('--bos_idx', type=int, default=0,
                    help='the index of <bos>.')
parser.add_argument('--eos_idx', type=int, default=0,
                    help='the index of <eos>.')
parser.add_argument('--pad_idx', type=int, default=0,
                    help='the index of <pad>.')
parser.add_argument('--use_bn', type=int, default=0,
                    help='whether to use batch normalization.')
parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='the dropout rate of the output layer.')
# for Relational Memory
parser.add_argument('--rm_num_slots', type=int, default=3,
                    help='the number of memory slots.')
parser.add_argument('--rm_num_heads', type=int, default=8,
                    help='the numebr of heads in rm.')
parser.add_argument('--rm_d_model', type=int,
                    default=512, help='the dimension of rm.')

# Sample related
parser.add_argument('--sample_method', type=str, default='beam_search',
                    help='the sample methods to sample a report.')
parser.add_argument('--beam_size', type=int, default=3,
                    help='the beam size when beam searching.')
parser.add_argument('--temperature', type=float,
                    default=1.0, help='the temperature when sampling.')
parser.add_argument('--sample_n', type=int, default=1,
                    help='the sample number per image.')
parser.add_argument('--group_size', type=int,
                    default=1, help='the group size.')
parser.add_argument('--output_logsoftmax', type=int,
                    default=1, help='whether to output the probabilities.')
parser.add_argument('--decoding_constraint', type=int,
                    default=0, help='whether decoding constraint.')
parser.add_argument('--block_trigrams', type=int,
                    default=1, help='whether to use block trigrams.')
hparams = parser.parse_args()
args = c(hparams)
hparams.args = args


class ERbert(nn.Module):
    def __init__(self, encoder, decoder_path):
        super(ERbert, self).__init__()


        self.encoder = encoder
        self.encoder.reset_head(num_classes=512)
        self.encoder_tokenizer = encoder.tokenizer
        self.text_encoder = encoder.lm_model
        self.proj_t = encoder.proj_t
        # setup decoder
        # self.encoder = VisualExtractor(args, encoder)
        self.tokenizer = self.setup_tokenizer(decoder_path)
        self.encoder_decoder = EncoderDecoder(args, self.tokenizer)
        self.loss = compute_loss

    # def __str__(self):
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def setup_tokenizer(self, decoder_path):
        tokenizer = BertTokenizerFast.from_pretrained(
            decoder_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        return self.add_special_all_special_tokens(tokenizer)
    
    def prepare_decoder_input(self, report):
        '''
        Create decoder input ids, attention mask and label ids.
        '''
        
        report = [self.tokenizer.bos_token + i + self.tokenizer.eos_token for i in report]
        tokenizer_output = self.tokenizer.batch_encode_plus(report,
                                                                add_special_tokens=True,
                                                                truncation=True,
                                                                max_length=256+1,
                                                                padding='max_length',
                                                                return_tensors='pt')
        
        decoder_input_ids = torch.tensor(tokenizer_output["input_ids"]).contiguous().clone()
        decoder_attention_mask = torch.tensor(tokenizer_output["attention_mask"])[:, 1:].clone()
        # label_ids = decoder_input_ids[:, 1:].detach().clone()
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_input_ids[decoder_input_ids ==
                            self.tokenizer.sep_token_id] = self.tokenizer.pad_token_id

        return decoder_input_ids, decoder_attention_mask#, label_ids
    
    def encoder_forward(self, ecg):
        att_feats, fc_feats = self.encoder.forward_encoding(ecg)
       
        return att_feats.to(ecg.device), fc_feats.to(ecg.device)
    
    def forward(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.encoder_forward(images)

        if mode == 'train':
            output = self.encoder_decoder(
                fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(
                fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
    
        return output

    def training_step(self, batch):
        output = self(batch['images'],
                      batch['decoder_input_ids'], mode='train')
        loss = self.loss(
            output, batch['decoder_input_ids'], batch['decoder_attention_mask'])
        # loss = loss.item()
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True,
                      prog_bar=True, batch_size=batch['images'].size(0), sync_dist=True)

        return loss
    

    def generate(self, images):
        """
        Autoregressively generate a prediction.
        Note that beam_size will be called automatically via args.
        """
        outputs = self(images, mode='sample')
        generated_report = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        return generated_report 
    

    @staticmethod
    def add_special_all_special_tokens(tokenizer):
        tokenizer.add_special_tokens({'bos_token': '[BOS]',
                                      'eos_token': '[EOS]',
                                      #   'pad_token': '[PAD]'
                                      })
        return tokenizer    