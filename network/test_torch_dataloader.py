# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:14:51 2018

@author: Administrator
"""
import torch
import torch.utils.data as Data

BATCH_SIZE = 5      # 批训练的数据个数
 
x = torch.linspace(1, 10, 10)       # x data (torch tensor)
y = torch.linspace(10, 1, 10)       # y data (torch tensor)
 
# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(x,y)
 
# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)
 