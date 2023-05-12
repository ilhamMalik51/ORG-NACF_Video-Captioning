'''
Module :  config
Author:  Nasibullah (nasibullah104@gmail.com)
Details : Ths module consists of all hyperparameters and path details corresponding to Different models and datasets.
          Only changing this module is enough to play with different model configurations. 
Use : Each model has its own configuration class which contains all hyperparameter and other settings. once Configuration object is created, use it to create Path object.
          
'''

import torch
import os

class Path:
    '''
    Currently supports MSVD and MSRVTT
    VATEX will be added in future
    '''
    def __init__(self, cfg, working_path):

        if cfg.dataset == 'msvd':   
            self.local_path = os.path.join(working_path,'MSVD')     
            self.video_path = 'path_to_raw_video_data' # For future use
            self.caption_path = os.path.join(self.local_path,'captions')
            self.feature_path = os.path.join(self.local_path,'features')
        
            self.name_mapping_file = os.path.join(self.caption_path,'youtube_mapping.txt')
            self.train_annotation_file = os.path.join(self.caption_path,'sents_train_lc_nopunc.txt')
            self.val_annotation_file = os.path.join(self.caption_path,'sents_val_lc_nopunc.txt')
            self.test_annotation_file = os.path.join(self.caption_path,'sents_test_lc_nopunc.txt')
            
          
            if cfg.appearance_feature_extractor == 'inceptionv4':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSVD_APPEARANCE_INCEPTIONV4_28.hdf5')
                
            if cfg.appearance_feature_extractor == 'inceptionresnetv2':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSVD_APPEARANCE_INCEPTIONRESNETV2_28.hdf5')
                
            if cfg.appearance_feature_extractor == 'resnet101':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSVD_APPEARANCE_RESNET101_28.hdf5')
                
            if cfg.appearance_feature_extractor == 'resnet101hc':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSVD_APPEARANCE_RESNET101_HC.hdf5')
            
            self.motion_feature_file = os.path.join(self.feature_path,'MSVD_MOTION_RESNEXT101.hdf5')
            #self.object_feature_file = os.path.join(self.feature_path,'MSVD_APPEARANCE_INCEPTIONV4.hdf5')
                

        if cfg.dataset == 'msrvtt':
            self.local_path = os.path.join(working_path,'MSRVTT')
            self.video_path = '/media/nasibullah/Ubuntu/DataSets/MSRVTT/'
            self.caption_path = os.path.join(self.local_path, 'captions')
            self.feature_path = os.path.join(self.local_path, 'features')
            
            self.category_file_path = os.path.join(self.caption_path, 'category.txt')
            self.train_val_annotation_file = os.path.join(self.caption_path, 'train_val_videodatainfo.json')
            self.test_annotation_file = os.path.join(self.caption_path, 'test_videodatainfo.json')
            
            if cfg.appearance_feature_extractor == 'inceptionv4':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSRVTT_APPEARANCE_INCEPTIONV4_28.hdf5')
                
            if cfg.appearance_feature_extractor == 'inceptionresnetv2':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSRVTT_APPEARANCE_INCEPTIONRESNETV2_28.hdf5')
                
            if cfg.appearance_feature_extractor == 'resnet101':
                self.appearance_feature_file = os.path.join(self.feature_path,'MSRVTT_APPEARANCE_RESNET101_28.hdf5')

            self.motion_feature_file = os.path.join(self.feature_path,'MSRVTT_MOTION_RESNEXT101.hdf5')
            self.object_feature_file = os.path.join(self.feature_path,'MSRVTT_OBJECT_FASTERRCNN_RESNEXT_28.hdf5')

            self.val_id_list = list(range(6513,7010))
            self.train_id_list = list(range(0,6513))
            self.test_id_list = list(range(7010,10000))

        self.prediction_path = 'results'
        self.saved_models_path = 'Saved'
        
            
class ConfigORGTRL:
    '''
    Hyperparameter settings for Soft Attention based LSTM (SA-LSTM) model.
    '''
    def __init__(self, 
                 model_name='sa-lstm', 
                 opt_encoder=True):
        
        self.model_name = model_name
        
        #Device configuration
        self.cuda_device_id = 0
        if torch.cuda.is_available():
            self.device = torch.device('cuda:' + str(self.cuda_device_id)) 
        else:
            self.device = torch.device('cpu')
        
        #Dataloader configuration
        self.dataset = 'msvd' 
        self.batch_size = 32 #suitable 
        self.val_batch_size = 10
        self.opt_truncate_caption = True
        self.max_caption_length = 24
        
        
        # Encoder configuration
        self.appearance_feature_extractor = 'inceptionresnetv2'
        self.motion_feature_extractor = 'resnext101'
        self.frame_len = 28
        self.motion_depth = 16  
        self.appearance_input_size = 1536
        self.appearance_projected_size = 512
        self.motion_input_size = 2048
        self.motion_projected_size = 512
        # New Configuration
        self.object_input_size = 1024
        self.object_projected_size = 512
        self.object_kernel_size = (1, 1)
        self.opt_encoder = opt_encoder
        
        
        #Decoder configuration
        self.decoder_type = 'lstm'
        self.embedding_size = 300 # word embedding size using gloVe
        if self.opt_encoder:
            self.feat_size =  self.appearance_projected_size
        else:
            self.feat_size = self.appearance_input_size 
        
        self.feat_size = self.appearance_projected_size
        # self.decoder_input_size = self.appearance_projected_size + self.motion_projected_size + self.embedding_size
        self.decoder_hidden_size = 512  #Hidden size of decoder LSTM
        self.attn_size = 512  # attention bottleneck

        # this size reflects the concatenation operation of
        # mean pooled global vide features, previous language lstm hidden state,
        # and previous word embedding
        # this implementation STILL USE CONTEXT GLOBAL AND ATTENTION LSTM HIDDEN STATE
        # self.attention_lstm_input_size = (self.appearance_projected_size * 2) + self.decoder_hidden_size + self.embedding_size
        # CEK THE INPUT SIZE
        self.attention_lstm_input_size = self.appearance_projected_size + self.decoder_hidden_size + self.embedding_size
        
        # this size reflects the concatenation operation of
        # context global, context local, and attentionLSTM hidden state
        # this implementation STILL USE CONTEXT GLOBAL AND ATTENTION LSTM HIDDEN STATE
        # CEK THE INPUT SIZE
        self.language_lstm_input_size = self.appearance_projected_size + self.object_projected_size + self.decoder_hidden_size 

        self.n_layers = 1
        self.embed_dropout = 0.5
        self.rnn_dropout = 0.4
        self.dropout = 0.5
        self.input_dropout = 0.2
        self.opt_param_init = False
        self.beam_length = 5
        
        #Training configuration
        self.encoder_lr = 3e-4
        self.decoder_lr = 3e-4
        self.teacher_forcing_ratio = 1.0 #
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.print_every = 400
        
        self.lr_decay_start_from = 20
        self.lr_decay_gamma = 0.5
        self.lr_decay_patience = 5
        self.weight_decay = 1e-5
        self.reg_lambda = 0.

        #Vocabulary configuration
        self.SOS_token = 1
        self.EOS_token = 2
        self.PAD_token = 0
        self.UNK_token = 3
        self.vocabulary_min_count = 2   
       