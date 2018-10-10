# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 08:45:52 2018

@author: Administrator
"""


import pandas as pd
import talib as ta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# import sys

# local_path = "C://Users//Administrator/Desktop/"
# if local_path not in sys.path:
#     sys.path.append(local_path)

import data_reader as rd
import tictoc as tim
import numpy as np
import matplotlib.pyplot as plt




# weight_for_50 = pd.read_csv("C:/Users/Administrator/Desktop/sz50const.csv", index_col=0)
import pymssql
conn=pymssql.connect(               #connect
        	server='192.168.0.28', port=1433,
        	user='sa', password='abc123',
        	database='rawdata', charset='utf8'
        	)
SQL = 'SELECT * FROM dbo.[50成分]'
weight_for_50 = pd.read_sql(SQL, conn, index_col='0')

stock_pool = set(weight_for_50.values[0])

for i in weight_for_50.values:
    stock_pool = stock_pool.intersection(set(i))

print("Phase I : Gathering data")
open_data = rd.get_muti_open_day(stock_pool, "2010-01-01", "2019-01-01")
print("Open data fetched")
close_data = rd.get_muti_close_day(stock_pool, "2010-01-01", "2019-01-01")
print("Close data fetched")
high_data = rd.get_muti_high_day(stock_pool, "2010-01-01", "2019-01-01")
low_data = rd.get_muti_low_day(stock_pool, "2010-01-01", "2019-01-01")
vol_data = rd.get_muti_vol_day(stock_pool, "2010-01-01", "2019-01-01")
print("Phase I : Done ")


close_data = close_data / close_data.iloc[0]
open_data = open_data / open_data.iloc[0]
high_data = high_data / high_data.iloc[0]
low_data = low_data / low_data.iloc[0]
vol_data = vol_data / vol_data.iloc[0]


close_data["benchmark_close"] = close_data.mean(axis=1)
open_data["benchmark_open"] = open_data.mean(axis=1)
high_data["benchmark_high"] = high_data.mean(axis=1)
low_data["benchmark_low"] = low_data.mean(axis=1)
vol_data["benchmark_vol"] = (vol_data).sum(axis=1)


benchmark = pd.DataFrame(index=close_data.index)

benchmark = pd.concat(
    [
        benchmark,
        close_data["benchmark_close"],
        open_data["benchmark_open"],
        high_data["benchmark_high"],
        low_data["benchmark_low"],
        vol_data["benchmark_vol"],
    ],
    axis=1,
)

time_data = rd.get_index_day("000001.SH", "2010-01-01", "2019-01-01", "1D")


close_data = close_data / close_data.iloc[0]

stock_pool = close_data.columns[:-1]

TRAIN_SIZE = 1200
TIME_SERIES = 150
BATCH_SIZE = 200
TRAIN_DATE = "2015-01-01"
LAYER = 2
HOLDING = 1

holding_return = (close_data.shift(-HOLDING) - close_data) / close_data
holding_return = holding_return.dropna()

"""
pytorch
"""


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, LAYER, batch_first=True)
        self.linear = nn.Linear(input_size, 10)
        self.hidden = self.initHidden()
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(10, output_size)
        self.nolinear = nn.Tanh()

    def forward(self, input):

        out, self.hidden = self.lstm(input, self.hidden)
        out = out[:, -1, :]
        out = self.nolinear(self.linear(out))
        out = self.linear2(out)
        return out

    def initHidden(self):

        return (
            torch.randn(LAYER, BATCH_SIZE, self.input_size).cuda(),
            torch.randn(LAYER, BATCH_SIZE, self.hidden_size).cuda(),
        )

    def pre_hidden(self):

        return (
            torch.randn(LAYER, 1, self.input_size).cuda(),
            torch.randn(LAYER, 1, self.hidden_size).cuda(),
        )


class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, output_size), nn.Tanh())
        self.decoder = nn.Sequential(nn.Tanh(), nn.Linear(output_size, input_size))

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return encoded, decoded


def label(x):
    if x >= 11:
        return 1
    else:
        return 0


label = np.vectorize(label)


def create_ta_data(stock):
    print("Processing", stock)
    if stock in stock_pool:
        SS = rd.get_stock_day(stock, "2010-01-01", "2019-01-01", "1D")
        SS = SS / SS.iloc[0]
        Return = holding_return[stock]
        target = holding_return.rank(axis=1)[stock]

        xlabel = label(target)
        target = pd.DataFrame(target)
        target["label"] = xlabel
        target = target[["label"]]

    elif stock == "benchmark":
        SS = benchmark
        SS.columns = ["sclose", "sopen", "high", "low", "vol"]

    MA = pd.DataFrame()
    # 计算10-90日均线
    for i in range(10, 50, 10):
        local_MA = ta.MA(SS.sclose, timeperiod=i)
        local_MA.name = stock + "_MA_" + str(i)
        MA = pd.concat([MA, local_MA], axis=1)
    MACD1, MACD2, XX = ta.MACD(SS.sclose)
    MACD = pd.concat([MACD1, MACD2], axis=1)
    MACD.columns = [stock + "_MACD1", stock + "_MACD2"]
    ADX = ta.ADX(SS.high, SS.low, SS.sclose) / 25
    ADXR = ta.ADXR(SS.high, SS.low, SS.sclose) / 25
    aroondown, aroonup = ta.AROON(SS.high, SS.low)
    aroondown = aroondown / 100
    aroonup = aroonup / 100
    ATR = ta.ATR(SS.high, SS.low, SS.sclose)
    Bupper, Bmiddle, Blower = ta.BBANDS(SS.sclose)
    group1 = pd.concat(
        [ADX, ADXR, aroondown, aroonup, ATR, Bupper, Bmiddle, Blower], axis=1
    )
    group1.columns = [
        stock + "_ADX",
        stock + "_ADXR",
        stock + "_aroondown",
        stock + "_aroonup",
        stock + "_ATR",
        stock + "_Bupper",
        stock + "_Bmiddle",
        stock + "_Blower",
    ]

    BOP = ta.BOP(SS.sopen, SS.high, SS.low, SS.sclose)
    CCI = ta.CCI(SS.high, SS.low, SS.sclose)
    CCI = CCI / 100
    CMO = ta.CMO(SS.sclose)
    CMO = CMO / 100
    DX = ta.DX(SS.high, SS.low, SS.sclose)
    EMA = ta.EMA(SS.sclose)
    KAMA = ta.KAMA(SS.sclose)
    MFI = ta.MFI(SS.high, SS.low, SS.sclose, SS.vol)
    MFI = MFI / 100
    MOM = ta.MOM(SS.sclose)
    RSI = ta.RSI(SS.sclose)
    RSI = (RSI - 50) / 100
    group2 = pd.concat([BOP, CCI, CMO, DX, EMA, KAMA, MFI, MOM, RSI], axis=1)
    group2.columns = [
        stock + "_BOP",
        stock + "_CCI",
        stock + "_CMO",
        stock + "_DX",
        stock + "_EMA",
        stock + "_KAMA",
        stock + "_MFI",
        stock + "_MOM",
        stock + "_RSI",
    ]
    SAR = ta.SAR(SS.high, SS.low)
    TRANGE = ta.TRANGE(SS.high, SS.low, SS.sclose)
    TRIMA = ta.TRIMA(SS.sclose)
    group3 = pd.concat([SAR, TRANGE, TRIMA], axis=1)
    group3.columns = [stock + "_SAR", stock + "_TRANGE", stock + "_TRIMA"]
    SS.columns = [
        stock + "_close",
        stock + "_open",
        stock + "_high",
        stock + "_low",
        stock + "_vol",
    ]
    raw_ta = pd.concat([SS, MA, MACD, group1, group2, group3], axis=1)
    if stock in stock_pool:
        raw_ta = pd.concat([raw_ta, target, Return], axis=1)

    return raw_ta


train_dict = {}
predict_dict = {}
benchmark_data = create_ta_data("benchmark")
for i in stock_pool:
    ta_data = create_ta_data(i)
    ta_data = pd.concat([benchmark_data, ta_data], axis=1)
    ta_data = ta_data.dropna()
    features = ta_data.iloc[:, :-2] #去掉倒数两列，分别是 label 和 Return
    features = (features - features.cummin()) / (features.cummax() - features.cummin()) # 归一化所有 features
    ta_data.iloc[:, :-2] = features
    ta_data = ta_data.dropna()
    train_dict[i] = ta_data[:TRAIN_SIZE]
    # TODO 为什么train_dict和predict_dict有部分重合？
    predict_dict[i] = ta_data.iloc[TRAIN_SIZE - TIME_SERIES :]


def train_data_collecter(train_dict):
    '''
    将输入的 DataFrame 数据打包转换为 tensor 对象
    '''
    train_xx = []
    train_y = []

    for key in train_dict.keys():
        stock_df = train_dict[key].copy()

        for i in range(TIME_SERIES, len(stock_df)):
            x = stock_df.iloc[i - TIME_SERIES : i, :-2].values
            x = x.reshape(TIME_SERIES, -1)
            train_xx.append(x)
            train_y.append(stock_df.iloc[i, -2])

    train_xx = np.array(train_xx)
    train_xx = torch.from_numpy(train_xx)
    train_y = np.array(train_y)
    train_y = torch.from_numpy(train_y)
    train_AE = train_xx
    AE_dataset = Data.TensorDataset(train_xx, train_AE)
    stock_dataset = Data.TensorDataset(train_xx, train_y)
    train_loader = Data.DataLoader(
        dataset=stock_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )
    AE_loader = Data.DataLoader(
        dataset=AE_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )
    return train_loader, AE_loader


train_loader, AE_loader = train_data_collecter(train_dict)
predict_loader, __ = train_data_collecter(predict_dict)


"""
LSTM v1.1

layer 1：  Linear（模仿PCA）    39  features ->  15 features 
layer 2:   LSTM（时间序列学习）  50  time series -> f 
layer 3:   LSTM（第二层）         
layer 4：  Linear（模仿PCA）    15  features ->   10 features
layer 5:   Dropout(控制过拟合)  50%
layre 6:   Linear(概率输出)     10 feature   ->   5 

"""

#%%
autoencoder = AutoEncoder(62, 32)
autoencoder = autoencoder.cuda()


def AE_featrure():

    loss_fn = nn.MSELoss()
    opt = optim.Adam(autoencoder.parameters(), lr=0.05)

    tim.tic()
    for epoch in range(50):
        total_loss = 0
        for orgin_data, data in AE_loader:
            encoded, decoded = autoencoder(orgin_data.float().cuda())
            loss = loss_fn(decoded, data.float().cuda())

            total_loss += loss.cpu().data.numpy()

            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            print(
                "AutoEncoder cycle "
                + str(epoch)
                + " done. The mean loss is "
                + str(total_loss / len(train_loader) / BATCH_SIZE)
            )
            tim.toc()
            tim.tic()
    return autoencoder.encoder

#等待AE_feature学习完毕以后，把学习的结果抽取出来，作为压缩原始数据维度的一种方式
decoded_layer = AE_featrure()


#%%
def eval_train():
    cnt = 0
    for feature, target in train_loader:
        if feature.size()[0] != BATCH_SIZE:
            continue
        mylstm.hidden = mylstm.initHidden()
        feature = feature.float().cuda()
        feature = decoded_layer(feature)
        # reduced_feature = decoded_layer(feature)
        target = target.long().cuda()
        out = mylstm(feature)
        p = out.cpu().data.numpy().argmax(axis=1)
        # print(p) #发现p全都是[1, 1, ..., 1]
        target = target.cpu().data.numpy()
        for i in range(len(p)):
            if p[i] == target[i]:
                cnt += 1
    print(cnt / len(train_loader) / 200)


def eval_predict():
    cnt = 0
    for feature, target in predict_loader:
        print(feature.shape)
        if feature.size()[0] != BATCH_SIZE:
            continue
        mylstm.hidden = mylstm.initHidden()
        feature = feature.float().cuda()
        feature = decoded_layer(feature)
        # reduced_feature = decoded_layer(feature)
        target = target.long().cuda()
        out = mylstm(feature)
        p = out.cpu().data.numpy().argmax(axis=1)
        print(p) #发现p全都是[1, 1, ..., 1]
        target = target.cpu().data.numpy()
        for i in range(len(p)):
            if p[i] == target[i]:
                cnt += 1
    print(cnt / len(predict_loader) / 200)


mylstm = LSTM(32, 32, 2)
mylstm = mylstm.cuda()
loss_fn = nn.CrossEntropyLoss().float().cuda()
opt = optim.Adam(mylstm.parameters(), lr=0.01)

collect = []
tim.tic()
print("start training...")
for epoch in range(501):
    print('epoch',epoch)
    total_loss = 0
    for feature, target in train_loader:
        if feature.size()[0] != BATCH_SIZE:
            continue
        mylstm.hidden = mylstm.initHidden()
        feature = feature.float().cuda()
        feature = decoded_layer(feature)
        target = target.long().cuda()
        out = mylstm(feature)
        loss = 0
        loss = loss_fn(out, target)
        total_loss += loss.cpu().data.numpy()
        opt.zero_grad()
        loss.backward()
        opt.step()

    if epoch > 0 and epoch % 10 == 0:
        print("样本内效果")
        eval_train()
        print("样本外效果")
        eval_predict()

    if epoch % 10 == 0:
        print("cycle " + str(epoch) + " done. The mean loss is " + str(total_loss / len(train_loader) / BATCH_SIZE))
        if total_loss / len(train_loader) / BATCH_SIZE < 0.0025:
            break
        tim.toc()
        tim.tic()
    collect.append(total_loss / len(train_loader) / BATCH_SIZE)

