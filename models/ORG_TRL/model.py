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
     
class ORG(nn.Module):
    def __init__(self, cfg):
        super(ORG, self).__init__()
        '''
        Object Relational Graph (ORG) is a module that learns 
        to describe an object based on its relationship 
        with others in a video.
        
        Arguments:
            feat_size : The object feature size that obtained from
                        the last fully-connected layer of the backbone
                        of Faster R-CNN
        '''
        # dropout rate 0.3
        self.dropout = nn.Dropout(cfg.input_dropout)

        # dropout rate 0.5
        self.att_weight_dropout = nn.Dropout(cfg.dropout)

        self.object_projection = nn.Linear(in_features=cfg.object_input_size,
                                           out_features=cfg.object_projected_size)

        self.sigma_r = nn.Linear(in_features=cfg.object_input_size,
                                 out_features=cfg.object_projected_size)
        
        self.psi_r = nn.Linear(in_features=cfg.object_input_size,
                               out_features=cfg.object_projected_size)
        
        self.w_r = nn.Linear(in_features=cfg.object_input_size, 
                             out_features=cfg.object_projected_size, 
                             bias=False)
    
    def forward(self, object_variable):
        '''
        input:
          r_feats: The embedded feature into 512-D which has shape of
                   (batch_size, feature_dim, num_frames, num_objects)
        output:
          r_hat: Enhanced features that has the information of relation
                 between objects 
                 (batch_size, num_frames, num_objects, feature_dim)
        '''
        object_variable = self.dropout(object_variable)

        ## Projected Object Features to 512-D
        r_feat = self.object_projection(object_variable)

        ## Sigma(R) = R . Wi + bi
        ## Psi(R) = R . Wj + bj
        ## A = Simga(R) . Psi(R).T
        a_coeff = torch.bmm(self.sigma_r(object_variable).view(-1, r_feat.size(-2), r_feat.size(-1)), 
                            self.psi_r(object_variable).contiguous().view(-1, r_feat.size(-2), r_feat.size(-1))\
                            .transpose(1, 2)).view(r_feat.size(0), r_feat.size(1), r_feat.size(-2), r_feat.size(-2))
        
        ## A_hat = Softmax(A)
        a_hat = F.softmax(a_coeff, dim=-1)

        ## Applying Dropout
        a_hat = self.att_weight_dropout(a_hat) 

        ## R_hat = A_hat . R . Wr
        r_hat = torch.matmul(a_hat, self.w_r(object_variable))
        
        return r_feat, r_hat


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
        # dropout rate 0.3
        self.dropout = nn.Dropout(cfg.input_dropout)

        self.v_projection_layer = nn.Linear(cfg.appearance_input_size + cfg.motion_input_size, #(batch_size, n_frames, 3600)
                                            cfg.appearance_projected_size)

        self.org_module = ORG(cfg)  
        
    def forward(self, appearance_feat, motion_feat, object_feat):
        
        appearance_feat = self.dropout(appearance_feat)
        motion_feat = self.dropout(motion_feat)

        v_feats = torch.cat([appearance_feat, motion_feat], dim=-1)

        ## jointly representation 512-D
        v_feats = self.v_projection_layer(v_feats)
        
        ## r_feats (batch_size, dim_feature, height, width)
        r_feats, r_hat = self.org_module(object_feat)

        return v_feats, r_feats, r_hat
    

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

class SpatialAttention(nn.Module):
    def __init__(self, cfg):
        super(SpatialAttention, self).__init__()
        '''
        Spatial Attention module. 
        It depends on previous hidden attention memory in the decoder attention,
        and the size of object features.  
        Argumets:
          decoder_hidden_size : hidden memory size of decoder. (batch, hidden_size)
          feat_size : feature size of object features.
          bottleneck_size : intermediate size.
        '''

        self.hidden_size = cfg.decoder_hidden_size
        self.feat_size = cfg.object_projected_size
        self.bottleneck_size = cfg.attn_size
        
        self.decoder_projection = nn.Linear(self.hidden_size,
                                            self.bottleneck_size,
                                            bias=False)
        self.encoder_projection = nn.Linear(self.feat_size, 
                                            self.bottleneck_size, 
                                            bias=False)
        self.energy_projection = nn.Linear(self.bottleneck_size, 
                                           1,
                                           bias=False)
     
    def forward(self, h_attn_lstm, obj_feats):
        '''
        shape of hidden (hidden_size) (batch,hidden_size) #(128, 512)
        shape of obj_feats (batch size, num_objects, feat_size)  #(128, 5, 512)
        '''
        Wv = self.encoder_projection(obj_feats)
        Uh = self.decoder_projection(h_attn_lstm)
        Uh = Uh.unsqueeze(1).expand_as(Wv)

        beta = F.softmax(self.energy_projection(torch.tanh(Wv + Uh)), dim=1)        
        global_context_feature = torch.sum(torch.mul(obj_feats, beta), dim=1)

        return global_context_feature

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

        self.attention_lstm = nn.LSTM(input_size=cfg.attention_lstm_input_size, 
                                      hidden_size=cfg.decoder_hidden_size,
                                      num_layers=self.n_layers, 
                                      dropout=self.rnn_dropout)
        
        self.temporal_attention = TemporalAttention(cfg)

        self.spatial_attention = SpatialAttention(cfg)

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
                v_features, 
                aligned_objects):
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
        v_features = self.embedding_dropout(v_features)
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
        last_hidden_attn = self.input_feature_dropout(last_hidden_attn)

        # context global vector
        # is the product of element-wise multiplication of
        # attention weight and v_features (batch_size, features_size * 2)
        context_global_vector, alpha = self.temporal_attention(last_hidden_attn, v_features)
        
        aligned_objects = self.embedding_dropout(aligned_objects)

        local_aligned_features = torch.sum(torch.mul(aligned_objects, alpha.unsqueeze(-1)), dim=1)

        # input local_aligned_features into the spatial attention
        context_local_vector = self.spatial_attention(last_hidden_attn, local_aligned_features)

        # concat [c_global, c_local, h_attn] (3, 128, 512) (L, N, dim_feature)
        input_lang_lstm = torch.cat((context_global_vector.unsqueeze(0), 
                                     context_local_vector.unsqueeze(0), 
                                     h_attn_lstm[0]), dim=-1)

        output, h_lang_lstm = self.language_lstm(input_lang_lstm, lang_hidden) # (1, 100, 512)
        
        output = output.squeeze(0) # (batch_size, features_From LSTM) (100, 512)
        output = self.embedding_dropout(output) # for regularization
        output = self.out(output) # (batch_size, vocabulary_size) (100, num_words)
        output = F.softmax(output, dim = 1) # In Probability Value (batch_size, vocabulary_size) (100, num_words)
        
        return output, h_attn_lstm, h_lang_lstm
    
    
class ORG_TRL(nn.Module):
    def __init__(self, 
                 voc, 
                 cfg, 
                 path):
        super(ORG_TRL,self).__init__()

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
        
        
    def load(self, encoder_path = 'Save/ORG_Encoder_10.pt', decoder_path='Saved/ORG_Decoder_10.pt'):
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.decoder.load_state_dict(torch.load(decoder_path))
        else:
            print('File not found Error..')

    def save(self, encoder_path, decoder_path):
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            torch.save(self.encoder.state_dict(),encoder_path)
            torch.save(self.decoder.state_dict(),decoder_path)
        else:
            print('Invalid path address given.')
    
    def align_object_variable(self, r_feats, r_hat):
        '''
        align object modul according to ORG-TRL Paper
        refers = https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Object_Relational_Graph_With_Teacher-Recommended_Learning_for_Video_Captioning_CVPR_2020_paper.pdf
        args:
        object_variable : This is object features exctracted from Faster RCNN
        output:
        aligned_object_variable
        '''
        ## Mengambil anchor frame sebagai acuan untuk setiap objek
        ## Memisahkan anchor frame dari keseluruhan fitur objek
        anchor_frame = r_feats[:, 0]
        next_frame = r_feats[:, 1:r_feats.size(1)]
        
        ## menghitung cosine similarity scores
        ## matmul( achor_frame, next_frame ) / | anchor_frame | * | next_frame |
        similarity_score = (torch.matmul(anchor_frame.unsqueeze(1), next_frame.transpose(2, -1)) / \
                            (torch.norm(anchor_frame.unsqueeze(1), dim=-1)[:, :, :, None] * \
                            torch.norm(next_frame, dim=-1)[:, :, None, :]))

        aligned_frames = torch.gather(r_hat[:, 1:r_hat.size(1)], 
                                      dim=2, 
                                      index=similarity_score.topk(1, -1)[1].\
                                      expand(-1, -1, -1, r_hat.size(-1)))

        return torch.cat([r_hat[:, 0].unsqueeze(1), aligned_frames], dim=1)
            
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
            appearance_features, targets, mask, max_length, _, motion_features, object_features = data
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            
            ## Di sini dicari index untuk aligned objects

            loss = self.train_iter(utils, 
                                   appearance_features,
                                   motion_features,
                                   object_features, 
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
                   object_variable,
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
        object_variable = object_variable.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.byte().to(self.device)
        
        if self.cfg.opt_encoder:
            v_features, r_feats, r_hat = self.encoder(input_variable, motion_variable, 
                                                      object_variable)

        aligned_objects = self.align_object_variable(r_feats, r_hat)  
        
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
                                                                                        v_features, 
                                                                                        aligned_objects)
                
                # Teacher forcing: next input comes from ground truth(data distribution)
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output.unsqueeze(0), 
                                                      target_variable[t], 
                                                      mask[t], 
                                                      self.device)
                loss += mask_loss
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
            appearance_features, targets, mask, max_length, _, motion_features, object_features = data
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            
            loss = self.eval_iter(utils, 
                                  appearance_features,
                                  motion_features,
                                  object_features, 
                                  targets, 
                                  mask, 
                                  max_length, 
                                  use_teacher_forcing)
            print_loss += loss
            total_loss += loss

        return total_loss/len(dataloader)
        
    @torch.no_grad()
    def eval_iter(self, utils, input_variable, motion_variable, object_variable,
                  target_variable, mask, max_target_len, use_teacher_forcing):
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
        object_variable = object_variable.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.byte().to(self.device)
        
        if self.cfg.opt_encoder:
            v_features, r_feats, r_hat = self.encoder(input_variable, motion_variable, 
                                                      object_variable)

        aligned_objects = self.align_object_variable(r_feats, r_hat)
        
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
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden_attn, decoder_hidden_lang = self.decoder(decoder_input,
                                                                                        decoder_hidden_attn,
                                                                                        decoder_hidden_lang, 
                                                                                        v_features, 
                                                                                        aligned_objects)
                
                # Teacher forcing: next input comes from ground truth(data distribution)
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output.unsqueeze(0), 
                                                      target_variable[t], 
                                                      mask[t], 
                                                      self.device)
                loss += mask_loss
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
                     object_feats, 
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
            v_features, r_feats, r_hat = self.encoder(feats, motion_feats, object_feats)   

        aligned_objects = self.align_object_variable(r_feats, r_hat)

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
                                                                          v_features,
                                                                          aligned_objects) ## NEED TO CHECK

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
    
   
