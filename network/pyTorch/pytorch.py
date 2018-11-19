# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:45:10 2018

@author: Administrator
"""
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms

N_SAMPLES = 20
N_HIDDEN = 300

x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x+0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()


net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)
 
net_dropped = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net_dropped.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()
 
plt.ion()
plt.show()

for t in range(500):
    pred_ofit = net_overfitting(x)
    pred_drop = net_dropped(x)
 
    loss_ofit = loss_func(pred_ofit, y)
    loss_drop = loss_func(pred_drop, y)
 
    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    loss_ofit.backward()
    loss_drop.backward()
    optimizer_ofit.step()
    optimizer_drop.step()
    if t % 50 == 0:     # 每 10 步画一次图
        # 将神经网络转换成测试形式, 画好图之后改回 训练形式
        net_overfitting.eval()
        net_dropped.eval()  # 因为 drop 网络在 train 的时候和 test 的时候参数不一样.
        plt.cla()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)
        input = torch.linspace(-1, 1, N_SAMPLES)
        input = torch.unsqueeze(input,dim=1)
        outputofit = net_overfitting(input)
        outputdrop = net_dropped(input)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
        plt.plot(input.data.numpy(),outputdrop.data.numpy(),'b--',label='dropout')
        plt.plot(input.data.numpy(),outputofit.data.numpy(),'r',label='ovfit')
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)
 
        # 将两个网络改回 训练形式
        net_overfitting.train()
        net_dropped.train()