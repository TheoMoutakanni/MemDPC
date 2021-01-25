import math
import numpy as np
import sys
import copy
sys.path.append('../')
from backbone.convrnn import ConvGRU

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(1, '../../TimeCycle/models/videos')
import resnet_res4s1
import inflated_resnet


class CycleTime(nn.Module):
    def __init__(self, sample_size, num_seq, seq_len, dropout=0.5, num_class=101, train_what='all'):
        super(CycleTime, self).__init__()

        resnet = resnet_res4s1.resnet18()#pretrained=pretrained)
        self.encoderVideo = inflated_resnet.InflatedResNet(copy.deepcopy(resnet))

       	torch.cuda.manual_seed(666)
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.num_class = num_class 
        self.train_what = train_what

        self.lstm = nn.LSTM(256, 256, num_layers=1, batch_first=True)

        self.final_bn = nn.BatchNorm1d(256)
        self.final_bn.weight.data.fill_(1)
        self.final_bn.bias.data.zero_()

        self.final_fc = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(256, self.num_class))
        self._initialize_weights(self.final_fc)

    def forward(self, block):
        # seq1: [B, N, C, SL, W, H]
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        enable_grad = self.train_what!='last'
        with torch.set_grad_enabled(enable_grad):
	        x = self.encoderVideo(block)
	        x = F.avg_pool3d(x, (5, 16, 16), stride=1).squeeze(-1).squeeze(-1)

	        x = F.normalize(x, p=2, dim=1)
	        x = x.view(B, N, 256)

        _, (context, _) = self.lstm(x)
        context = self.final_bn(context.squeeze()) #.transpose(-1,-2)).transpose(-1,-2) # [B,N,C] -> [B,C,N] -> BN() -> [B,N,C], because BN operates on id=1 channel.
        output = self.final_fc(context) #.view(B, -1, self.num_class)

        return output, context

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)        


