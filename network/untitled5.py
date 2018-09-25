# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:02:37 2018

@author: Administrator
"""

import pandas as pd
import talib as ta
import torch
import torch.nn as nn
import torch.optim as optim
import sys
local_path = 'C://Users//Administrator/Desktop/'
if local_path not in sys.path:
    sys.path.append(local_path)
from get_data import data_reader as rd
from get_data import tic_toc

tim = tic_toc.tic()


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(LSTM, self).__init__()
        self.input_size =input_size
        self.hidden_size = hidden_size
        self.lstm_layer1 = nn.LSTM(input_size,hidden_size)
        self.linear = nn.Linear(input_size,output_size)
        self.hidden = self.initHidden()
        self.drop = nn.Dropout(0.6)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, input):
        
        out,self.hidden = self.lstm(input,self.hidden)
        out = self.drop(out[-1])
        out = self.linear(out)
        out = self.softmax(out)
        return out

    def initHidden(self):
        
        return (torch.randn(1,1,self.input_size).cuda(),torch.randn(1,1,self.hidden_size).cuda())