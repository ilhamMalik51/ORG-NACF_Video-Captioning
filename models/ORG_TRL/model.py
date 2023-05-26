'''
Module :  ORG-TRL Model
Modified by :  Muhammad Ilham Malik (muh.ilham.malik@gmail.com)
'''


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.transforms as transforms

import random
import itertools
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import itertools

import numpy as np
import os
import copy

from transformers import BertForMaskedLM, DistilBertForMaskedLM, MobileBertForMaskedLM
     
class InputEmbeddingLayer(nn.Module):
    def __init__(self, cfg):
        super(InputEmbeddingLayer, self).__init__()
        self.w1 = nn.Linear(cfg.encoder_input_size, cfg.encoder_projected_size)
        self.w2 = nn.Linear(cfg.encoder_projected_size, cfg.encoder_projected_size)
        self.w3 = nn.Linear(cfg.encoder_projected_size, cfg.encoder_projected_size)
        self.dropout = nn.Dropout(cfg.encoder_dropout_size)
        
    def forward(self, feats):
        v_bar = self.w1(feats)
        v_hat = torch.tanh(self.w2(v_bar))
        gate = torch.sigmoid(self.w3(v_hat))
        result = gate * v_bar + (1 - gate) * v_hat
        
        return self.dropout(result)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder,self).__init__()
        '''
        Encoder module. Project the video feature into a different space which will be 
        send to decoder.
        Argumets:
          input_size : CNN extracted feature size. For ResNet 2048, For inceptionv4 1536
          output_size : Dimention of projected space.
        '''
        self.iel_appearance_feature = InputEmbeddingLayer(cfg)
        self.iel_motion_feature = InputEmbeddingLayer(cfg)
         
        
    def forward(self, appearance_feat, motion_feat):
        appearance = self.iel_appearance_feature(appearance_feat)
        motion = self.iel_motion_feature(motion_feat)

        return torch.cat([appearance, motion], dim=1)
    

class TemporalAttention(nn.Module):
    def __init__(self, cfg):
        super(TemporalAttention, self).__init__()
        '''
        Temporal Attention Module of ORG.
        It depends on previous hidden state of LSTM attention.
        Arguments:
          lstm_attn_hidden: The hidden state from LSTM attention
                            tensors of shape (batch_size, hidden_size).
          video_feats_size: The concatenation of frame features
                            and motion features.
                            tensors of shape (batch_size, n_frames, feats_size)
          attn_size       : The attention size of attention module.
        '''

        self.hidden_size = cfg.decoder_hidden_size
        self.features_size = cfg.feat_size
        self.attn_size = cfg.attn_size
        
        # the input of concatenated features has size of 512
        self.encoder_projection = nn.Linear(self.features_size, 
                                            self.attn_size, 
                                            bias=False)
        
        self.decoder_projection = nn.Linear(self.hidden_size, 
                                            self.attn_size, 
                                            bias=False)
        
        self.energy_projection = nn.Linear(self.attn_size, 
                                           1, 
                                           bias=False)
     
    def forward(self, h_attn_lstm, v_features):
        '''
        shape of hidden attention lstm (batch_size, hidden_size)
        shape of video features input (batch_size, n_frames, features_size)
        '''

        Wv = self.encoder_projection(v_features)
        Uh = self.decoder_projection(h_attn_lstm)
        Uh = Uh.unsqueeze(1).expand_as(Wv)

        alpha = F.softmax(self.energy_projection(torch.tanh(Wv + Uh)),  dim=1)
        context_global = torch.sum(torch.mul(alpha, v_features), dim=1)

        return context_global, alpha
    

class DecoderRNN(nn.Module):
    def __init__(self, cfg, voc):
        super(DecoderRNN, self).__init__()
        '''
        Decoder, Basically a language model.
        Args:
        hidden_size : hidden memory size of LSTM/GRU
        output_size : output size. Its same as the vocabulary size.
        n_layers : 
        
        '''
        
        # Keep for reference
        self.dropout = cfg.dropout
        self.feat_len = cfg.frame_len
        self.attn_size = cfg.attn_size
        self.output_size = voc.num_words
        self.rnn_dropout = cfg.rnn_dropout
        self.n_layers = cfg.n_layers
        self.decoder_type = cfg.decoder_type

        # Define layers
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(voc.gloVe_embedding).float(),
            freeze=False,
            padding_idx=0)
        
        self.embedding_dropout = nn.Dropout(self.dropout)

        # CHECK NEW
        # self.embedding = nn.Embedding(voc.num_words, cfg.embedding_size)

        # CHECK NEW
        '''
        Attention LSTM Module.
          inputs: Has inputs shape of 1324. This size comes from the concat
                  operation of global-mean-pooled video features,
                  previous language lstm hidden state, and
                  previous word embedding.
          hidden_size : according to the ORG-TRL paper the hidden state is 512
        '''
        self.attention_lstm = nn.LSTM(input_size=cfg.attention_lstm_input_size, 
                                      hidden_size=cfg.decoder_hidden_size,
                                      num_layers=self.n_layers, 
                                      dropout=self.rnn_dropout)
        
        # CHECK NEW
        self.temporal_attention = TemporalAttention(cfg)

        self.embedding_dropout = nn.Dropout(cfg.dropout)

        self.language_lstm = nn.LSTM(input_size=cfg.language_lstm_input_size, 
                                     hidden_size=cfg.decoder_hidden_size,
                                     num_layers=self.n_layers, 
                                     dropout=self.rnn_dropout)
            
        self.out = nn.Linear(cfg.decoder_hidden_size, self.output_size)

    
    def forward(self,
                inputs, 
                attn_hidden,
                lang_hidden, 
                v_features):
        '''
        we run this one step (word) at a time
        
        inputs -  (1, batch)
        hidden - h_n/c_n :(num_layers * num_directions, batch, hidden_size)    
        GRU:h_n   
        LSTM:(h_n,c_n)
        feats - (batch,attention_length,annotation_vector_size) 
        
        '''
        '''
        NEW CHECK

        inputs -  (1, batch)
        hidden - h_n/c_n :(num_layers * num_directions, batch, hidden_size)    
        GRU:h_n   
        LSTM:(h_n,c_n)
        feats - (batch, attention_length, annotation_vector_size) 
        '''

        embedded = self.embedding(inputs) # [inputs:(1, batch)  outputs:(1, batch, embedding_size)]
        embedded = self.embedding_dropout(embedded)

        # global mean pooled the v features
        v_bar_features = torch.mean(v_features, dim=1, keepdim=True).squeeze(1).unsqueeze(0)
        
        # preparing the input for lstm
        # concat [v_bar, word_emb, h_lang_prev] shape(512 + 300 + 512)
        input_attn_lstm = torch.cat((v_bar_features, embedded, lang_hidden[0]), dim=-1)

        # h_attn_lstm have the shape of (hidden, cell)
        # hidden has the shape of (n_layer, batch_size, hidden_size)
        _, h_attn_lstm = self.attention_lstm(input_attn_lstm, attn_hidden)
        
        # get the last hidden layer
        last_hidden_attn = h_attn_lstm[0] if self.decoder_type=='lstm' else h_attn_lstm
        last_hidden_attn = last_hidden_attn.view(self.n_layers, 
                                                 last_hidden_attn.size(1), 
                                                 last_hidden_attn.size(2))
        last_hidden_attn = last_hidden_attn[-1]

        # context global vector
        # is the product of element-wise multiplication of
        # attention weight and v_features (batch_size, features_size * 2)

        context_global_vector, alpha = self.temporal_attention(last_hidden_attn, v_features)

        # concat [c_global, c_local, h_attn] (3, 128, 512) (L, N, dim_feature)
        input_lang_lstm = torch.cat((context_global_vector.unsqueeze(0),
                                     h_attn_lstm[0]), dim=-1)

        output, h_lang_lstm = self.language_lstm(input_lang_lstm, lang_hidden) # (1, 100, 512)
        
        output = output.squeeze(0) # (batch_size, features_From LSTM) (128, 512)
        output = self.out(output) # (batch_size, vocabulary_size) (128, num_words)
        output = F.softmax(output, dim = 1) # In Probability Value (batch_size, vocabulary_size) (128, num_words)
        
        return output, h_attn_lstm, h_lang_lstm
    
    
class ORG_TRL(nn.Module):
    def __init__(self, 
                 voc, 
                 cfg, 
                 path):
        super(ORG_TRL, self).__init__()

        self.voc = voc
        self.path = path
        self.cfg = cfg
        
        if cfg.opt_encoder:
            self.encoder = Encoder(cfg).to(cfg.device)
            self.enc_optimizer = optim.Adam(self.encoder.parameters(), 
                                            lr=cfg.encoder_lr)

        self.decoder = DecoderRNN(cfg, voc).to(cfg.device)
        self.dec_optimizer = optim.Adam(self.decoder.parameters(), 
                                        lr=cfg.decoder_lr, 
                                        weight_decay=cfg.weight_decay, 
                                        amsgrad=True)
    
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        self.print_every = cfg.print_every
        self.clip = cfg.clip
        self.device = cfg.device

        if cfg.opt_param_init:
            self.init_params()
        
    def index_from_bert(self, x):
        return self.cfg.bert2index.get(x)
        
    def init_params(self):
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
                #nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
   
        
    def update_hyperparameters(self,cfg):
        if self.cfg.opt_encoder:
            self.enc_optimizer = optim.Adam(self.encoder.parameters(), lr=cfg.encoder_lr)
        
        self.dec_optimizer = optim.Adam(self.decoder.parameters(), lr=cfg.decoder_lr, amsgrad=True)
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        
        
    def load(self, encoder_path = 'Save/ORG_Encoder_10.pt', decoder_path = 'Saved/ORG_Decoder_10.pt'):
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.decoder.load_state_dict(torch.load(decoder_path))
        else:
            print('File not found Error..')

    def save(self, encoder_path, decoder_path):
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            torch.save(self.encoder.state_dict(), encoder_path)
            torch.save(self.decoder.state_dict(), decoder_path)
        else:
            print('Invalid path address given.')

    ## NEW TRL TRAINING PARADIGM
    def set_trl(self, bert_variant="Bert"):

        if bert_variant == "Bert":
            bert_model_path='bert_finetuned_1000_data_fix.pt'
            index2bert_path='index2mbert_emb.pt'
            bert2index_path='mbert2index_emb.pt'
            
            bert_pth = os.path.join('Saved', bert_model_path)
            i2b = os.path.join('Saved', index2bert_path)
            b2i = os.path.join('Saved', bert2index_path)

            if os.path.exists(bert_pth):
                self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

                if self.device == 'cpu':
                    self.bert_model.load_state_dict(torch.load(bert_pth, map_location=torch.device('cpu')))
                else:
                    self.bert_model.load_state_dict(torch.load(bert_pth))
                    print('Bert Model Loaded to GPU')

                self.bert_model = self.bert_model.to(self.device)
                self.index2bert = torch.load(i2b).to(self.device).requires_grad_(False)
                self.bert2index = torch.load(b2i).to(self.device).requires_grad_(False)

                print('Fine-tuned Bert Model and Index2Bert Tensor Loaded Successfully')
            else:
                print('File not found Error..')
        
        elif bert_variant == "MobileBert":
            bert_model_path='MobileBert_finetuned_train_val_data.pt'
            index2bert_path='index2mbert_emb.pt'
            bert2index_path='mbert2index_emb.pt'
            
            bert_pth = os.path.join('Saved', bert_model_path)
            i2b = os.path.join('Saved', index2bert_path)
            b2i = os.path.join('Saved', bert2index_path)

            if os.path.exists(bert_pth):
                self.bert_model = MobileBertForMaskedLM.from_pretrained('google/mobilebert-uncased')

                if self.device == 'cpu':
                    self.bert_model.load_state_dict(torch.load(bert_pth, map_location=torch.device('cpu')))
                else:
                    self.bert_model.load_state_dict(torch.load(bert_pth))
                    print('MobileBert Model Loaded to GPU')

                self.bert_model = self.bert_model.to(self.device)
                self.index2bert = torch.load(i2b).to(self.device).requires_grad_(False)
                self.bert2index = torch.load(b2i).to(self.device).requires_grad_(False)

                print('Fine-tuned MobileBert Model and Index2Bert Tensor Loaded Successfully')
            else:
                print('File not found Error..')

        elif bert_variant == "DistilBert":
            bert_model_path='DistilBert_finetuned_train_val_data.pt'
            index2bert_path='index2dbert_emb.pt'
            bert2index_path='dbert2index_emb.pt'
            
            bert_pth = os.path.join('Saved', bert_model_path)
            i2b = os.path.join('Saved', index2bert_path)
            b2i = os.path.join('Saved', bert2index_path)

            if os.path.exists(bert_pth):
                self.bert_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

                if self.device == 'cpu':
                    self.bert_model.load_state_dict(torch.load(bert_pth, map_location=torch.device('cpu')))
                else:
                    self.bert_model.load_state_dict(torch.load(bert_pth))
                    print('DistilBert Model Loaded to GPU')

                self.bert_model = self.bert_model.to(self.device)
                self.index2bert = torch.load(i2b).to(self.device).requires_grad_(False)
                self.bert2index = torch.load(b2i).to(self.device).requires_grad_(False)

                print('Fine-tuned DistilBert Model and Index2DistilBert Tensor Loaded Successfully')
                print("dbert2ndex embedding shape: {}".format(self.bert2index.shape))
                print("index2dbert embedding shape: {}".format(self.index2bert.shape))
            else:
                print('File not found Error..')
    
            
    def train_epoch(self,
                    dataloader,
                    utils):
        '''
        Function to train the model for a single epoch.
        Args:
         Input:
            dataloader : the dataloader object.basically train dataloader object.
         Return:
            epoch_loss : Average single time step loss for an epoch
        '''
        total_loss = 0
        start_iteration = 1
        print_loss = 0
        iteration = 1

        if self.cfg.opt_encoder:
            self.encoder.train()
        self.decoder.train()

        for data in dataloader:
            ## Perlu Merubah Dataloader
            appearance_features, targets, mask, max_length, _, motion_features, _ = data
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

            loss = self.train_iter(utils, 
                                   appearance_features,
                                   motion_features,
                                   targets, 
                                   mask, 
                                   max_length, 
                                   use_teacher_forcing)
            print_loss += loss
            total_loss += loss

            # Print progress
            if iteration % self.print_every == 0:
                print_loss_avg = print_loss / self.print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".
                format(iteration, iteration / len(dataloader) * 100, print_loss_avg))
                print_loss = 0
             
            iteration += 1

        return total_loss/len(dataloader)
        
        
    def train_iter(self, 
                   utils,
                   input_variable,
                   motion_variable,
                   target_variable,
                   mask,
                   max_target_len,
                   use_teacher_forcing
                   ):
        '''
        Forward propagate input signal and update model for a single iteration. 
        
        Args:
        Inputs:
            input_variable : video mini-batch tensor; size = (B, T, F)
            target_variable : Ground Truth Captions;  size = (T, B);
            T will be different for different mini-batches
            mask : Masked tensor for Ground Truth;    size = (T, C)
            max_target_len : maximum lengh of the mini-batch; size = T
            use_teacher_forcing : binary variable. If True training uses teacher forcing else sampling.
            clip : clip the gradients to counter exploding gradient problem.
        Returns:
            iteration_loss : average loss per time step.
        '''
        if self.cfg.opt_encoder:
            self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()
        
        loss = 0
        print_losses = []
        n_totals = 0
        
        input_variable = input_variable.to(self.device)
        motion_variable = motion_variable.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.byte().to(self.device)
        
        if self.cfg.opt_encoder:
            v_features = self.encoder(input_variable, motion_variable)  
        
        # Forward pass through encoder
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(self.cfg.batch_size)]])
        decoder_input = decoder_input.to(self.device)

        decoder_hidden = torch.zeros(self.cfg.n_layers, 
                                     self.cfg.batch_size,
                                     self.cfg.decoder_hidden_size).to(self.device)
        
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden_attn = (decoder_hidden, decoder_hidden)
            decoder_hidden_lang = (decoder_hidden, decoder_hidden)
        
        # concat the input and motion variable
        # v_features = torch.cat((input_variable, motion_variable), dim=-1).to(self.device) 
        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden_attn, decoder_hidden_lang = self.decoder(decoder_input,
                                                                                        decoder_hidden_attn,
                                                                                        decoder_hidden_lang, 
                                                                                        v_features)
                
                # Teacher forcing: next input comes from ground truth(data distribution)
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output.unsqueeze(0), 
                                                      target_variable[t], 
                                                      mask[t], 
                                                      self.device)
                
                # Preparing input for bert token IDs
                bert_input = F.embedding(target_variable.T, self.index2bert).squeeze(-1)
                bert_input[:, t][bert_input[:, t] != 0] = 103

                with torch.no_grad():
                    bert_output = self.bert_model(bert_input)[0]
                    bert_output = F.softmax((bert_output / self.cfg.temperature), dim=-1)
                
                # ambil predicted word pada timestep t
                topk_bert_probs, topk_indices = torch.topk(bert_output[:, t, :], k=50, dim=-1)
                topk_indices = F.embedding(topk_indices, self.bert2index).squeeze(-1)
                
                mask_p = torch.ones_like(topk_indices)
                mask_p[(topk_indices.eq(3) | topk_indices.eq(0))] = 0 # 3 adalah Unknown Token dan 0 adalah Padding Token

                kl_loss = utils.KLLoss(torch.gather(decoder_output, 1, topk_indices), 
                                       topk_bert_probs,
                                       mask_p, 
                                       t)

                loss += (((1 - self.cfg.lambda_var) * mask_loss) + (self.cfg.lambda_var * kl_loss))
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden_attn, decoder_hidden_lang = self.decoder(decoder_input,
                                                                                        decoder_hidden_attn,
                                                                                        decoder_hidden_lang, 
                                                                                        v_features.float())
                
                # No teacher forcing: next input is decoder's own current output(model distribution)
                _, topi = decoder_output.squeeze(0).topk(1)

                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.cfg.batch_size)]])

                decoder_input = decoder_input.to(self.device)
                # Calculate and accumulate loss
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output, 
                                                      target_variable[t], 
                                                      mask[t],
                                                      self.device)
                
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        # Perform backpropatation
        loss.backward()

        if self.cfg.opt_encoder:
            _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
            self.enc_optimizer.step()
        
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        self.dec_optimizer.step()
        
        return sum(print_losses) / n_totals
    
    def eval_epoch(self,
                   dataloader,
                   utils):
        '''
        Function to train the model for a single epoch.
        Args:
         Input:
            dataloader : the dataloader object.basically train dataloader object.
         Return:
             epoch_loss : Average single time step loss for an epoch
        '''
        total_loss = 0

        print_loss = 0

        for data in dataloader:
            appearance_features, targets, mask, max_length, _, motion_features, _ = data
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            
            loss = self.eval_iter(utils, 
                                  appearance_features,
                                  motion_features, 
                                  targets, 
                                  mask, 
                                  max_length)
            print_loss += loss
            total_loss += loss
        return total_loss/len(dataloader)
        
    @torch.no_grad()
    def eval_iter(self, utils, input_variable, motion_variable,
                  target_variable, mask, max_target_len):
        '''
        Forward propagate input signal and update model for a single iteration. 
        Args:
        Inputs:
            input_variable : video mini-batch tensor; size = (B, T, F)
            target_variable : Ground Truth Captions;  size = (T, B);
            T will be different for different mini-batches
            mask : Masked tensor for Ground Truth;    size = (T, C)
            max_target_len : maximum lengh of the mini-batch; size = T
            use_teacher_forcing : binary variable. If True training uses teacher forcing else sampling.
            clip : clip the gradients to counter exploding gradient problem.
        Returns:
            iteration_loss : average loss per time step.

        '''

        loss = 0
        print_losses = []
        n_totals = 0
        
        input_variable = input_variable.to(self.device)
        motion_variable = motion_variable.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.byte().to(self.device)
        
        if self.cfg.opt_encoder:
            v_features = self.encoder(input_variable, motion_variable)
        
        # Forward pass through encoder
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(10)]])\
            .to(self.device)
        decoder_hidden = torch.zeros(self.cfg.n_layers, 10,
                                     self.cfg.decoder_hidden_size).to(self.device)
        
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden_attn = (decoder_hidden, decoder_hidden)
            decoder_hidden_lang = (decoder_hidden, decoder_hidden)
        
        # concat the input and motion variable
        # v_features = torch.cat((input_variable, motion_variable), dim=-1).to(self.device)        
        for t in range(max_target_len):
            decoder_output, decoder_hidden_attn, decoder_hidden_lang =\
                self.decoder(decoder_input, decoder_hidden_attn, decoder_hidden_lang, 
                             v_features)
            
            # No teacher forcing: next input is decoder's own current output(model distribution)
            _, topi = decoder_output.squeeze(0).topk(1)

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(10)]])

            decoder_input = decoder_input.to(self.device)            
            # Calculate and accumulate loss
            mask_loss, nTotal = utils.maskNLLLoss(decoder_output, target_variable[t], 
                                                  mask[t], self.device)
            
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
        
        return sum(print_losses) / n_totals

    @torch.no_grad()
    def GreedyDecoding(self, 
                       features, 
                       motion_features,
                       object_features, 
                       max_length=24
                       ):
        
        batch_size = features.size()[0]
        features = features.to(self.device)
        motion_features = motion_features.to(self.device)
        object_features = object_features.to(self.device)
        
        if self.cfg.opt_encoder:
            # features, motion_features = self.encoder(features, motion_features) #need to make optional
            v_features, r_obj_feats, r_hat = self.encoder(features, 
                                                          motion_features,
                                                          object_features)   
        
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(batch_size)]]).to(self.device)
        decoder_hidden = torch.zeros(self.cfg.n_layers, 
                                     batch_size,
                                     self.cfg.decoder_hidden_size).to(self.device)
        
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden = (decoder_hidden,decoder_hidden)

        caption = []
        attention_values = []

        for _ in range(max_length):
            decoder_output, decoder_hidden, attn_values = self.decoder(decoder_input, 
                                                                       decoder_hidden,
                                                                       features.float(),
                                                                       motion_features.float(),
                                                                       )
            _, topi = decoder_output.squeeze(0).topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).to(self.device)
            caption.append(topi.squeeze(1).cpu())
            attention_values.append(attn_values.squeeze(2))

        caption = torch.stack(caption, 0).permute(1, 0)
        caps_text = []

        for dta in caption:
            tmp = []
            for token in dta:
                if token.item() not in self.voc.index2word.keys() or token.item()==2: # Remove EOS and bypass OOV
                    pass
                else:
                    tmp.append(self.voc.index2word[token.item()])
            tmp = ' '.join(x for x in tmp)
            caps_text.append(tmp)

        return caption, caps_text, torch.stack(attention_values,0).cpu().numpy()
    

    @torch.no_grad()
    def BeamDecoding(self, 
                     feats, 
                     motion_feats,
                     width, 
                     alpha=0., #This is a diversity parameter
                     max_caption_len=24
                     ):
        
        # inisialisasi variable
        batch_size = feats.size(0)
        vocab_size = self.voc.num_words

        # inisialisasi fungsi untuk merubah indeks ke kata
        vfunc = np.vectorize(lambda t: self.voc.index2word[t]) # to transform tensors to words
        rfunc = np.vectorize(lambda t: '' if t == 'EOS' else t) # to transform EOS to null string
        lfunc = np.vectorize(lambda t: '' if t == 'SOS' else t) # to transform SOS to null string
        pfunc = np.vectorize(lambda t: '' if t == 'PAD' else t) # to transform PAD to null string
        
        # penggunaan encoder
        if self.cfg.opt_encoder:
            v_features = self.encoder(feats, motion_feats)   

        # inisialisasi h_0 atau hidden state awal
        hidden = torch.zeros(self.cfg.n_layers, batch_size, self.cfg.decoder_hidden_size).to(self.device)

        # memeriksa apabila decoder yang digunakan adalah LSTM
        if self.cfg.decoder_type == 'lstm':
            hidden = (hidden, hidden)
        
        # inisialisasi variable-variable list
        input_list = [ torch.cuda.LongTensor(1, batch_size).fill_(self.cfg.SOS_token) ]

        # variabel ini digunakan untuk menyimpan hidden list
        # dari setiap top-k kata sebelumnya
        hidden_list_attn = [ hidden ]
        hidden_list_lang = [ hidden ]

        # list total probabilitas yang diperoleh
        # list untuk menyimpan nilai score
        cum_prob_list = [ torch.ones(batch_size).cuda() ]
        cum_prob_list = [ torch.log(cum_prob) for cum_prob in cum_prob_list ]

        # inisialisasi indeks end of sentence
        EOS_idx = self.cfg.EOS_token

        # menyimpat nilai output hasil dari decoder?
        output_list = [ [[]] for _ in range(batch_size) ]

        # penggabungan fitur
        # v_features = torch.cat((feats, motion_feats), dim=-1)
        
        for t in range(max_caption_len + 1):
            
            # inisialisasi variable
            # masih bingung fungsi variabel ini untuk apa
            beam_output_list = [] # width x ( 1, 100 )
            normalized_beam_output_list = [] # width x ( 1, 100 )
            
            if self.cfg.decoder_type == "lstm":
                # inisialisasi untuk menyimpan list hidden state
                beam_hidden_list_attn = ( [], [] ) # 2 * width x ( 1, 100, 512 )
                beam_hidden_list_lang = ( [], [] )
            else:
                beam_hidden_list = [] # width x ( 1, 100, 512 )
            
            # next output ?
            next_output_list = [ [] for _ in range(batch_size) ]
            
            # memeriksa semua panjang ukuran sama
            assert len(input_list) == len(hidden_list_attn) == len(hidden_list_lang) == len(cum_prob_list)
            
            for i, (input, h_attn, h_lang, cum_prob) in enumerate(zip(input_list, 
                                                                      hidden_list_attn, 
                                                                      hidden_list_lang, 
                                                                      cum_prob_list)):
                
                # output, next_hidden, _ = self.decoder(input, hidden, feats, motion_feats) # need to check

                output, next_hidden_attn, next_hidden_lang = self.decoder(input,
                                                                          h_attn,
                                                                          h_lang, 
                                                                          v_features) ## NEED TO CHECK

                caption_list = [ output_list[b][i] for b in range(batch_size)]
                EOS_mask = [ 0. if EOS_idx in [ idx.item() for idx in caption ] else 1. for caption in caption_list ]
                EOS_mask = torch.cuda.FloatTensor(EOS_mask)
                EOS_mask = EOS_mask.unsqueeze(1).expand_as(output)
                
                output = EOS_mask * output
                output += cum_prob.unsqueeze(1)
                beam_output_list.append(output)

                caption_lens = [ [ idx.item() for idx in caption ].index(EOS_idx) + 1 if EOS_idx in [ idx.item() for idx in caption ] else t + 1 for caption in caption_list ]
                caption_lens = torch.cuda.FloatTensor(caption_lens)

                normalizing_factor = ((5 + caption_lens) ** alpha) / (6 ** alpha)
                normalizing_factor = normalizing_factor.unsqueeze(1).expand_as(output)
                normalized_output = output / normalizing_factor
                normalized_beam_output_list.append(normalized_output)

                if self.cfg.decoder_type == "lstm":
                    beam_hidden_list_attn[0].append(next_hidden_attn[0])
                    beam_hidden_list_attn[1].append(next_hidden_attn[1])

                    beam_hidden_list_lang[0].append(next_hidden_lang[0])
                    beam_hidden_list_lang[1].append(next_hidden_lang[1])
                else:
                    beam_hidden_list_lang.append(next_hidden_lang)
                
                #end for i

            beam_output_list = torch.cat(beam_output_list, dim=1) # ( 100, n_vocabs * width )
            normalized_beam_output_list = torch.cat(normalized_beam_output_list, dim=1)
            beam_topk_output_index_list = normalized_beam_output_list.argsort(dim=1, descending=True)[:, :width] # ( 100, width )

            ## Floor division and modulus operation
            topk_beam_index = beam_topk_output_index_list // vocab_size # ( 100, width )
            topk_output_index = beam_topk_output_index_list % vocab_size # ( 100, width )

            ## Hal ini berarti akan memiliki ukuran (Batch, Width)
            topk_output_list = [ topk_output_index[:, i] for i in range(width) ] 
            
            if self.cfg.decoder_type == "lstm":
                topk_hidden_list_attn = (
                    [ [] for _ in range(width) ],
                    [ [] for _ in range(width) ]) # 2 * width * (1, 100, 512)
                
                topk_hidden_list_lang = (
                    [ [] for _ in range(width) ],
                    [ [] for _ in range(width) ])
            else:
                topk_hidden_list = [ [] for _ in range(width) ] # width * ( 1, 100, 512 )
            

            topk_cum_prob_list = [ [] for _ in range(width) ] # width * ( 100, )
            
            for i, (beam_index, output_index) in enumerate(zip(topk_beam_index, topk_output_index)):
                for k, (bi, oi) in enumerate(zip(beam_index, output_index)):
                    if self.cfg.decoder_type == "lstm":
                        topk_hidden_list_attn[0][k].append(beam_hidden_list_attn[0][bi][:, i, :])
                        topk_hidden_list_attn[1][k].append(beam_hidden_list_attn[1][bi][:, i, :])

                        topk_hidden_list_lang[0][k].append(beam_hidden_list_lang[0][bi][:, i, :])
                        topk_hidden_list_lang[1][k].append(beam_hidden_list_lang[1][bi][:, i, :])
                    else:
                        topk_hidden_list[k].append(beam_hidden_list[bi][:, i, :])

                    topk_cum_prob_list[k].append(beam_output_list[i][vocab_size * bi + oi])
                    next_output_list[i].append(output_list[i][bi] + [ oi ])
            
            output_list = next_output_list

            input_list = [ topk_output.unsqueeze(0) for topk_output in topk_output_list ] # width * ( 1, 100 )
            
            if self.cfg.decoder_type == "lstm":
                hidden_list_attn = (
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list_attn[0] ],
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list_attn[1] ]) # 2 * width * ( 1, 100, 512 )
                hidden_list_attn = [ ( hidden, context ) for hidden, context in zip(*hidden_list_attn) ]

                hidden_list_lang = (
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list_lang[0] ],
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list_lang[1] ]) # 2 * width * ( 1, 100, 512 )
                hidden_list_lang = [ ( hidden, context ) for hidden, context in zip(*hidden_list_lang) ]
            else:
                hidden_list = [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list ] # width * ( 1, 100, 512 )

            cum_prob_list = [ torch.cuda.FloatTensor(topk_cum_prob) for topk_cum_prob in topk_cum_prob_list ] # width * ( 100, )

            #end for t

        SOS_idx = self.cfg.SOS_token
        outputs = [ [ SOS_idx ] + o[0] for o in output_list ]
        
        outputs = [[torch.tensor(y) for y in x] for x in outputs]
        outputs = [[y.item() for y in x] for x in outputs]
        
        captions = vfunc(outputs)
        captions = rfunc(captions)
        captions = lfunc(captions)
        captions = pfunc(captions)
        caps_text = []

        for eee in captions:
            caps_text.append(' '.join(x for x in eee).strip())
        
        return caps_text