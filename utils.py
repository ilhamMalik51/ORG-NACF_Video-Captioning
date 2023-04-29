'''
Module :  utils
Author:  Nasibullah (nasibullah104@gmail.com)
          
'''

import sys
import os
import re
import random
import unicodedata
import numpy as np
import torch
sys.path.append('pycocoevalcap/')
sys.path.append('pycocoevalcap/bleu')
sys.path.append('pycocoevalcap/cider')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

class Utils:
    '''
    Generic utility functions that our model and dataloader would require

    '''
   
    @staticmethod
    def set_seed(seed):
        '''
          For reproducibility 
        '''
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def unicodeToAscii(s):
        '''
        Turn a Unicode string to plain ASCII, 
        Thanks to https://stackoverflow.com/a/518232/2809427
        '''
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def normalizeString(s):
        '''
        Lowercase, trim, and remove non-letter 
        '''
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    @staticmethod
    def target_tensor_to_caption(voc,target):
        '''
        Convert target tensor to Caption
        '''
        gnd_trh = []
        lend = target.size()[1]
        for i in range(lend):
            tmp = ' '.join(voc.index2word[x.item()] for x in target[:,i])
            gnd_trh.append(tmp)
        return gnd_trh

    @staticmethod
    def maskNLLLoss(inp, target, mask, device):
        '''
        Masked cross-entropy loss calculation; 
        refers: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
        '''
        inp = inp.squeeze(0)
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp.squeeze(0), 1, target.view(-1, 1)).squeeze(1).float())
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, nTotal.item()
    
    @staticmethod
    def KLLoss(decoder_output, bert_output, topk_indices, topk_bert_indices, t):
        '''
        Simplified Kullback-Leibler divergence loss
        refers: https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Object_Relational_Graph_With_Teacher-Recommended_Learning_for_Video_Captioning_CVPR_2020_paper.pdf
        '''
        mask = torch.ones_like(topk_indices)
        mask[(topk_indices.eq(3) | topk_indices.eq(0))] = 0 # 3 adalah Unknown Token dan 0 adalah Padding Token

        log_loss_p = torch.log(torch.gather(decoder_output, 1, topk_indices)) # perlu dikali mask karena unknown token tidak dihitung untuk loss
        log_loss_p = log_loss_p * mask
        proba_q = torch.gather(bert_output[:, t], 1, topk_bert_indices)
        kl_loss = (-(proba_q * log_loss_p).sum(dim=-1)).mean()
        return kl_loss
 
    @staticmethod
    def score(ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    @staticmethod
    def FrameCapture(video_path, video_name):
        '''
        Function to extract frames
        For MSVD Sample every 10th frame
        '''
        
        #video_path = video_path_dict[video_name]
        # Path to video file 
        video_path = video_path + video_name  #Changes
        vidObj = cv2.VideoCapture(video_path) 
        count = 0
        fail = 0
        # checks whether frames were extracted 
        success = 1
        frames = []
        while success: 
            # OpenCV Uses BGR Colormap
            success, image = vidObj.read() 
            try:
                RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if count%10 == 0:            #Sample 1 frame per 10 frames
                    frames.append(RGBimage)
                count += 1
            except:
                fail += 1
        vidObj.release()
        if count > 80:
            frames = frames[:81]
        return np.stack(frames[:-1]),count-1, fail

    @staticmethod
    def imshow(img):
        '''
        Shows a grid of images
        '''
        #img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

