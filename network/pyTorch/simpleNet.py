# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 12:20:53 2018

@author: Administrator
"""
import datetime 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#read_path = r'C:\Users\Administrator\Desktop'
#if read_path not in sys.path:
#    sys.path.append(read_path)

#from get_data import data_reader as rd

today = datetime.datetime.today()
today = str(today)[:10]

#ss = rd.get_index_day('000001.SH','2010-01-01',today,'1D')


class simpleNet(nn.Module):
    def __init__(self,hidden_size,hidden_size2):
        super(simpleNet, self).__init__()

        self.hidden_size = hidden_size

        self.layer1 = nn.Linear(1, hidden_size)
        self.layer2 = nn.Linear(hidden_size,hidden_size2)
        self.layer3 = nn.Linear(hidden_size2,1)
        self.ReLU = nn.ReLU()
        
    def forward(self, input):
        
        hidden = self.layer1(input)
        hidden = self.ReLU(hidden)
        hidden = self.layer2(hidden)
        hidden = self.ReLU(hidden)
        out = self.layer3(hidden)
        
        return out

loss_fun = torch.nn.MSELoss()
t = simpleNet(30,20)
opt = optim.SGD(t.parameters(),lr=0.01)
collect = []

x = torch.unsqueeze(torch.linspace(-np.pi, np.pi, 100), dim = 1)
x = x.view(10,1,10)
y = torch.sin(x) + 0.1 * torch.rand(x.size())
x=x.cuda()
y=y.cuda()
t = t.cuda()

tim.tic()
for i in range(50):
    total_loss = 0
    for batch in range(10):
        opt.zero_grad()
        out = t(x[batch].t())
        loss = loss_fun(out,y[batch].t())
        loss.backward()
        opt.step()
        total_loss += loss
    collect.append(total_loss/10)
tim.toc()