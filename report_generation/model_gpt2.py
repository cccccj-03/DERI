import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (GPT2Config, GPT2TokenizerFast,
                          GPT2LMHeadModel, PretrainedConfig, EncoderDecoderModel)
from transformers.modeling_outputs import BaseModelOutput


class ERGPT2(nn.Module):
    def __init__(self, encoder, decoder_path):
        super(ERGPT2, self).__init__()

        self.encoder = encoder
        self.encoder.reset_head(num_classes=768)
        config = GPT2Config.from_pretrained(decoder_path, output_hidden_states=True)
        config.add_cross_attention = True
        config.is_decoder = True
        config.output_hidden_states = True

        self.decoder_path = decoder_path
        decoder = GPT2LMHeadModel.from_pretrained(
                decoder_path, config=config)
        decoder.resize_token_embeddings(config.vocab_size + 2)
        print("Load pretrained distillgpt2")

        class DummyEncoder:
            main_input_name = 'dummy'

            class DummyConfig(PretrainedConfig):
                model_type = 'bert'

            config = DummyConfig()

            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size

            def get_output_embeddings(cls):
                return None
            
            def forward(cls):
                return cls
        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_decoder = EncoderDecoderModel(
                    encoder=dummy_encoder, decoder=decoder)
                
        self.decoder = Decoder()
        self.tokenizer = self.setup_tokenizer()

    def setup_tokenizer(self):
        # Decoder tokenizer:
        tokenizer = GPT2TokenizerFast.from_pretrained(self.decoder_path)
        tokenizer.add_special_tokens(
            {"bos_token": "[BOS]", 'pad_token': '[PAD]'})

        # Print the special tokens:
        print('Description, Special token, Index')
        for k, v in tokenizer.special_tokens_map.items():
            if k != 'additional_special_tokens':
                print(f'{k}, {v}, {getattr(tokenizer, k + "_id")}')
            else:
                for i, j in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids):
                    print(f'additional_special_token, {i}, {j}')
        return tokenizer

    def encoder_forward(self, ecg):

        ecg_features = self.encoder(ecg).unsqueeze(1)
        encoder_outputs = BaseModelOutput(last_hidden_state=ecg_features)

        return encoder_outputs


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
        
        decoder_input_ids = torch.tensor(tokenizer_output["input_ids"]).contiguous()
        decoder_attention_mask = torch.tensor(tokenizer_output["attention_mask"])[:, 1:]
        label_ids = decoder_input_ids[:, 1:].detach().clone()
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_input_ids[decoder_input_ids ==
                            self.tokenizer.sep_token_id] = self.tokenizer.pad_token_id

        return decoder_input_ids, decoder_attention_mask, label_ids

    def forward(self, ecg, report):

        encoder_outputs = self.encoder_forward(ecg)
        # Teacher forcing: labels are given as input

        input_ids, attention_mask, label_ids = self.prepare_decoder_input(report)
        outputs = self.decoder.encoder_decoder(
            decoder_input_ids=input_ids.to(ecg.device),
            decoder_attention_mask=attention_mask.to(ecg.device),
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )

        return outputs.logits, input_ids.to(outputs.logits.device), attention_mask.to(outputs.logits.device), label_ids.to(outputs.logits.device)
    

