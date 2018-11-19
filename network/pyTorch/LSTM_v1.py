# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 08:38:10 2018

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
import numpy as np
import matplotlib.pyplot as plt
tim = tic_toc.tic()


TRAIN_SIZE = 1700
TIME_SERIES = 150
BATCH_SIZE = 50


"""
自编码器：1层非线性
"""

class AutoEncoder(nn.Module):
    def __init__(self,input_size,output_size):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_size,output_size),
                nn.Tanh()
                )
        self.decoder = nn.Sequential(
                nn.Linear(output_size,input_size),
                nn.Tanh()
                )
    def forward(self,input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return encoded,decoded
    


class LSTM(nn.Module):
    def __init__(self, input_size,reduced_size,hidden_size,output_size):
        super(LSTM, self).__init__()
        self.pca_linear = nn.Linear(input_size,reduced_size)
        self.reduced_size =reduced_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(reduced_size,hidden_size,2,batch_first = True)
        self.linear = nn.Linear(reduced_size,10)
        self.hidden = self.initHidden()
        self.drop = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=0)
        self.linear2 = nn.Linear(10,output_size)
        self.nolinear = nn.Tanh()
        
    def forward(self, input):
        out = self.pca_linear(input)
        out,self.hidden = self.lstm(out,self.hidden)
        out = out[:,-1,:]
        out = self.nolinear(self.linear(out))
        out = self.linear2(out)
        return out

    def initHidden(self):
        
        return (torch.randn(2,BATCH_SIZE,self.reduced_size).cuda(),torch.randn(2,BATCH_SIZE,self.hidden_size).cuda())
    def pre_hidden(self):
        
        return (torch.randn(2,1,self.reduced_size).cuda(),torch.randn(2,1,self.hidden_size).cuda())


"""
技术指标计算模块
"""
SS = rd.get_index_day('000905.SH','2000-01-01','2018-07-01',freq = '1D')

target = []
pct = SS.sclose.pct_change().dropna()
v = pct.values

for i in v:
    
    if i>0.015:
        target.append(4)
        
    elif i<-0.015:
        
        target.append(0)

    elif i>0.005:
        target.append(3)
    elif i<-0.005:
        target.append(1)
    else:
        target.append(2)
target = pd.DataFrame(data=target,index = pct.index, columns =['target'])

MA = pd.DataFrame()
#计算10-90日均线
for i in range(10,90,10):
    local_MA = ta.MA(SS.sclose,timeperiod = i)
    local_MA.name = 'MA'+str(i)
    MA = pd.concat([MA,local_MA],axis=1)

MACD1,MACD2,XX = ta.MACD(SS.sclose)
MACD = pd.concat([MACD1,MACD2],axis=1)
ADX = ta.ADX(SS.high,SS.low,SS.sclose)
ADXR = ta.ADXR(SS.high,SS.low,SS.sclose)
aroondown,aroonup = ta.AROON(SS.high,SS.low)
ATR = ta.ATR(SS.high,SS.low,SS.sclose)
Bupper,Bmiddle,Blower = ta.BBANDS(SS.sclose)
group1 = pd.concat([SS,MA,MACD,ADX,ADXR,aroondown,aroonup,ATR,Bupper,Bmiddle,Blower],axis=1)

BOP = ta.BOP(SS.sopen,SS.high,SS.low,SS.sclose)
CCI = ta.CCI(SS.high,SS.low,SS.sclose)
CMO = ta.CMO(SS.sclose)
DEMA = ta.DEMA(SS.sclose)
DX = ta.DX(SS.high,SS.low,SS.sclose)
EMA = ta.EMA(SS.sclose)
KAMA = ta.KAMA(SS.sclose)
MFI = ta.MFI(SS.high,SS.low,SS.sclose,SS.volumn)
MOM = ta.MOM(SS.sclose)
RSI = ta.RSI(SS.sclose)
group2 = pd.concat([BOP,CCI,CMO,DEMA,DX,EMA,KAMA,MFI,MOM,RSI],axis=1)

SAR = ta.SAR(SS.high,SS.low)
TEMA = ta.TEMA(SS.sclose)
TRANGE = ta.TRANGE(SS.high,SS.low,SS.sclose)
TRIMA = ta.TRIMA(SS.sclose)
TRIX = ta.TRIX(SS.sclose)
group3 = pd.concat([SAR,TEMA,TRANGE,TRIMA,TRIX],axis=1)

raw_ta = pd.concat([group1,group2,group3],axis=1)


ta_index = pd.concat([raw_ta,target],axis=1)

ta_index = ta_index.dropna() #38输入 1输出 4类型

x = ta_index.values[:,-1]

category_tensor = ta_index.iloc[:,-1].values

"""
技术指标计算模块结束
"""

"""
自编码器降维  39 -> 12
"""
'''
autoencoder = AutoEncoder(39,12)
AC_loss_func = nn.MSELoss()
AC_opt = optim.Adam(autoencoder.parameters(),lr=0.01)
AC_collect = []
tim.tic()
autoencoder = autoencoder.cuda()
for epoch in range(500):
    AC_total_loss = 0
    AC_cnt=0
    AC_batch_loss = 0
    AC_xx=[]
    AC_tt=[]
    for i in range(TIME_SERIES,TRAIN_SIZE):
        AC_x = ta_index.iloc[i-TIME_SERIES:i,:-1].values
        AC_x = (AC_x-AC_x.mean(axis=0))/AC_x.std(axis=0)
        AC_x = AC_x.reshape(TIME_SERIES,-1)
        AC_x = AC_x[-1]
        AC_xx.append(AC_x)
        AC_tt.append(AC_x)
        AC_cnt +=1
        if AC_cnt ==BATCH_SIZE:
            AC_xx = np.array(AC_xx)
            AC_xx = torch.from_numpy(AC_xx)
            AC_cnt = 0
            AC_xx = AC_xx.cuda()
            encoded,decoded = autoencoder(AC_xx.float())
            AC_t = torch.from_numpy(np.array(AC_tt)).float().cuda()
            AC_loss = AC_loss_func(decoded,AC_t)
            AC_total_loss += AC_loss.cpu().data.numpy()
            AC_opt.zero_grad()
            AC_loss.backward()
            AC_opt.step()
            AC_xx=[]
            AC_tt=[]
    AC_collect.append(AC_total_loss/(TRAIN_SIZE - TIME_SERIES))
    if epoch % 10 == 0:
        print("AutoEncoder cycle "+str(epoch)+" done. The mean loss is "+str(AC_total_loss/(TRAIN_SIZE-TIME_SERIES)))
        tim.toc()
        tim.tic()
'''



"""
LSTM v1.1

layer 1：  Linear（模仿PCA）    39  features ->  15 features 
layer 2:   LSTM（时间序列学习）  50  time series -> f 
layer 3:   LSTM（第二层）         
layer 4：  Linear（模仿PCA）    15  features ->   10 features
layer 5-:   Dropout(控制过拟合)  50%
layre 6:   Linear(概率输出)     10 feature   ->   5 

"""

mylstm = LSTM(39,15,15,5)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,1,1]).float().cuda())
opt = optim.Adam(mylstm.parameters(),lr=0.01)

def do_predict(epoch):
    long_val =[1]
    short_val=[1]
    long_short=[1]
    signal_list =[1]
    for i in range(TRAIN_SIZE,len(ta_index)-1):
        Return = ta_index.sclose.iloc[i+1]/ta_index.sclose.iloc[i]
        mylstm.hidden = mylstm.pre_hidden()
        x = torch.from_numpy(ta_index.iloc[i-TIME_SERIES:i,:-1].values)
        x = (x-x.mean(dim=0))/x.std(dim=0)
        x = x.contiguous().view(1,TIME_SERIES,-1)
        x = x.cuda()
        out = mylstm(x.float())
        signal = out[0].cpu().data.numpy().argmax()
        signal_list.append(signal)
    
        if signal == 4:
            long_val.append(long_val[-1]*Return)
            long_short.append(long_short[-1]*Return)
            short_val.append(short_val[-1])
        elif signal ==3:
            long_val.append(long_val[-1]*Return)
            long_short.append(long_short[-1]*Return)
            short_val.append(short_val[-1])
        elif signal == 0:
            long_val.append(long_val[-1])
            long_short.append(long_short[-1]*(2-Return))
            short_val.append(short_val[-1]*(2-Return))
        elif signal == 1:
            long_val.append(long_val[-1])
            long_short.append(long_short[-1]*(2-Return))
            short_val.append(short_val[-1]*(2-Return))
        else:
            long_val.append(long_val[-1])
            long_short.append(long_short[-1])
            short_val.append(short_val[-1])
    benchmark = ta_index.sclose.iloc[TRAIN_SIZE:]/ta_index.sclose.iloc[TRAIN_SIZE]
    df = pd.DataFrame(data=np.array([long_val,short_val,long_short,benchmark,signal_list,ta_index.target[TRAIN_SIZE:]]).T,index = ta_index.index[TRAIN_SIZE:],columns = ['long_val','short_val','long_short','benchmark','signal','ture'])
    
    rval = df.iloc[:,:4]
    rval = rval/rval.iloc[0]
    
    rval.plot()
    plt.show()
    pr = df.iloc[:,4]
    tg = df.iloc[:,5]
    
    delta = pr-tg
    delta.hist(bins=9)
    plt.show()
    df.to_excel(r'C:\Users\Administrator\Desktop\network\lstm测试\lstm_v1.1+_zz500_150T_'+str(epoch)+'_round1.xlsx')
    
def do_train(epoch):
    long_val =[1]
    short_val=[1]
    long_short=[1]
    signal_list =[1]
    for i in range(TIME_SERIES,TRAIN_SIZE-1):
        Return = ta_index.sclose.iloc[i+1]/ta_index.sclose.iloc[i]
        mylstm.hidden = mylstm.pre_hidden()
        x = torch.from_numpy(ta_index.iloc[i-TIME_SERIES:i,:-1].values)
        x = (x-x.mean(dim=0))/x.std(dim=0)
        x = x.contiguous().view(1,TIME_SERIES,-1)
        x = x.cuda()
        out = mylstm(x.float())
        signal = out[0].cpu().data.numpy().argmax()
        signal_list.append(signal)
    
        if signal == 4:
            long_val.append(long_val[-1]*Return)
            long_short.append(long_short[-1]*Return)
            short_val.append(short_val[-1])
        elif signal ==3:
            long_val.append(long_val[-1]*Return)
            long_short.append(long_short[-1]*Return)
            short_val.append(short_val[-1])
        elif signal == 0:
            long_val.append(long_val[-1])
            long_short.append(long_short[-1]*(2-Return))
            short_val.append(short_val[-1]*(2-Return))
        elif signal == 1:
            long_val.append(long_val[-1])
            long_short.append(long_short[-1]*(2-Return))
            short_val.append(short_val[-1]*(2-Return))
        else:
            long_val.append(long_val[-1])
            long_short.append(long_short[-1])
            short_val.append(short_val[-1])
    benchmark = ta_index.sclose.iloc[TIME_SERIES:TRAIN_SIZE]/ta_index.sclose.iloc[TIME_SERIES]
    df = pd.DataFrame(data=np.array([long_val,short_val,long_short,benchmark,signal_list,ta_index.target[TIME_SERIES:TRAIN_SIZE]]).T,index = ta_index.index[TIME_SERIES:TRAIN_SIZE],columns = ['long_val','short_val','long_short','benchmark','signal','ture'])
    
    rval = df.iloc[:,:4]
    rval = rval/rval.iloc[0]
    
    rval.plot()
    plt.show()
    pr = df.iloc[:,4]
    tg = df.iloc[:,5]
    
    delta = pr-tg
    delta.hist(bins=9)
    plt.show()

mylstm = mylstm.cuda()

"""
TRAIN PART GPU
"""
collect = []
tim.tic()
for epoch in range(501):
    total_loss = 0
    cnt=0
    batch_loss = 0
    xx=[]
    tt=[]
    for i in range(TIME_SERIES,TRAIN_SIZE):
        mylstm.hidden = mylstm.initHidden()
        x = ta_index.iloc[i-TIME_SERIES:i,:-1].values
        x = (x-x.mean(axis=0))/x.std(axis=0)
        x = x.reshape(TIME_SERIES,-1)
        xx.append(x)
        tt.append(category_tensor[i])
        cnt+=1
        if cnt ==BATCH_SIZE:
            xx = np.array(xx)
            xx = torch.from_numpy(xx)
            cnt = 0
            xx = xx.cuda()
            out = mylstm(xx.float())
            t = torch.from_numpy(np.array(tt)).long().cuda()
            loss = loss_fn(out,t)
            total_loss += loss.cpu().data.numpy()
            opt.zero_grad()
            loss.backward()
            opt.step()
            xx=[]
            tt=[]
    if epoch>0  and epoch % 100 == 0:
        print('样本内效果')
        do_train(epoch)
        print('样本外效果')
        do_predict(epoch)
        
    if epoch % 10 == 0:
        print("cycle "+str(epoch)+" done. The mean loss is "+str(total_loss/(TRAIN_SIZE-TIME_SERIES)))
        tim.toc()
        tim.tic()
    collect.append(total_loss/(TRAIN_SIZE-TIME_SERIES))
"""
PREDICT PART
"""
