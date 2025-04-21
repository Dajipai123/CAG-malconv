from collections import deque
from collections import OrderedDict 

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


from LowMemConv import LowMemConvBase


def getParams():
    #Format for this is to make it work easily with Optuna in an automated fashion.
    #variable name -> tuple(sampling function, dict(sampling_args) )
    params = {
        'channels'     : ("suggest_int", {'name':'channels', 'low':32, 'high':1024}),
        'log_stride'   : ("suggest_int", {'name':'log2_stride', 'low':2, 'high':9}),
        'window_size'  : ("suggest_int", {'name':'window_size', 'low':32, 'high':256}),
        'layers'       : ("suggest_int", {'name':'layers', 'low':1, 'high':6}),
        'embd_size'    : ("suggest_int", {'name':'embd_size', 'low':4, 'high':64}),
    }
    return OrderedDict(sorted(params.items(), key=lambda t: t[0]))

def initModel(**kwargs):
    new_args = {}
    for x in getParams():
        if x in kwargs:
            new_args[x] = kwargs[x]
            
    return MalConvML(**new_args)


class MalConvML(LowMemConvBase):
    
    def __init__(self, out_size=2, channels=128, window_size=512, stride=512, layers=1, embd_size=8, log_stride=None, device=None):
        super(MalConvML, self).__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        if not log_stride is None:
            stride = 2**log_stride
        
        # convs to extract features
        self.convs = nn.ModuleList([nn.Conv1d(embd_size, channels*2, window_size, stride=stride, bias=True)] + [nn.Conv1d(channels, channels*2, window_size, stride=1, bias=True) for i in range(layers-1)])
        #one-by-one cons to perform information sharing
        # 用于信息共享的一对一卷积, 用于将不同的卷积层的输出进行信息共享   
        self.convs_1 = nn.ModuleList([nn.Conv1d(channels, channels, 1, bias=True) for i in range(layers)])
        self.device = device
        
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
        
    
    def processRange(self, x):
        '''
            将输入数据 x 通过一系列的卷积层和激活函数，以提取其特征。
        '''
        # x = self.embd(x.to(self.device))
        x = self.embd(x.to('cuda:0'))

        #x = torch.transpose(x,-1,-2)
        # 重新排列张量的维度，并确保结果在内存中是连续的
        x = x.permute(0,2,1).contiguous()
        
        for conv_glu, conv_share in zip(self.convs, self.convs_1):
            x = F.leaky_relu(conv_share(F.glu(conv_glu(x.contiguous()), dim=1)))
        
        return x
    
    def forward(self, x):
        post_conv = x = self.seq2fix(x)
        
        penult = x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        
        return x, penult, post_conv
