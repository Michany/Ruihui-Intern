# -*- coding: utf-8 -*-
"""
@author: Michael Wang

反转因子
----------------

根据东吴证券研报《反转因子的精细结构》编写
2018/12/19
TODO

"""

import datetime
import time
from data_reader import get_index_day, get_muti_close_day

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn as skl
import talib
from sklearn.linear_model import LinearRegression

import pymssql

N = 60
year_start = 2010
year_end = 2017
data = get_stock_day('600000.SH','2010-01-01','2017-12-31')
# %% 选股
def collect_data():
    data = pd.DataFrame()
    
    return data 

def 当月最后一个交易日(data):
    dateList = []
    for year in range(year_start, year_end+1):
        for month in range(1,13):
            string = str(year)+'-'+str(month)
            dateList.append(data[string].index[-1])
    return dateList
        
def M_high(data):
    R = (1+data['sclose'].pct_change())
    for i in range(60, len(data)+1): # i代表行数，i从1开始计数，所以在index[i]时需要减1
        if data.index[i-1] in (dateList):# 月末不一定是当月交易日最后一天
            this = data['vol'].iloc[i-60:i]
            rank = this.rank()
            yield data.index[i-1], R.where(rank>30).fillna(1).cumprod().iloc[-1]-1
def M_low(data):
    R = (1+data['sclose'].pct_change())
    for i in range(60, len(data)+1): # i代表行数，i从1开始计数，所以在index[i]时需要减1
        if data.index[i-1] in (dateList):# 月末不一定是当月交易日最后一天
            this = data['vol'].iloc[i-60:i]
            rank = this.rank()#axis=1
            yield data.index[i-1], R.where(rank<=30).fillna(1).cumprod().iloc[-1]-1

def Ret(series, parameter = 20):
    return series.pct_change(periods=parameter)

    
dateList=当月最后一个交易日(data)
Mhigh = pd.DataFrame([row for row in M_high(data)], columns = ['date','Mhigh'])
Mlow = pd.DataFrame([row for row in M_low(data)], columns = ['date','Mlow'])
Ret20 = Ret(data, 20)

#factors = pd.concat([Mhigh.set_index('date'), Mlow.set_index('date')],axis=1)














#%% 获取价格数据
price = get_muti_close_day(selection.columns, '2009-03-31', '2018-11-30', freq = 'M', adjust=-1) # 回测时还是使用前复权价格
priceFill = price.fillna(method='ffill') 
price_change = priceFill.diff()
hs300 = get_index_day('000300.SH','2009-4-30','2018-11-30','M').sclose
szzz = get_index_day('000001.SH','2009-4-30','2018-11-30','M').sclose
zz500 = get_index_day('000905.SH','2009-4-30','2018-11-30','M').sclose
IC = pd.read_hdf("monthlyData-over10B.h5",'IC')
IF = pd.read_hdf("monthlyData-over10B.h5",'IF')


#%% 股票选择
isBig = (mv.T > mv.T.quantile(0.8)).T
isSmall = (mv.T < mv.T.quantile(0.3)).T
def reindex_fill_gap(inputDataFrame, defaultIndex = price.index):
    return inputDataFrame.reindex(defaultIndex, method='ffill')
selection = reindex_fill_gap(selection)
isBig = reindex_fill_gap(isBig)
isSmall = reindex_fill_gap(isSmall)

#%% 回测
pos = (mv.T/mv.T.sum()).T #按照市值加权作为仓位
pos = pos.reindex(price.index, method='ffill')

pos = pos[selection>0]# 选取selection中结果大于零的

大盘股=0
if 大盘股:
    pos = pos[isBig==True]# 选取大盘股
    pos = (pos.T/(pos.T.sum())).T# 将仓位调整至100%

#%% 计算当日盈亏百分比
daily_pnl = pos * priceFill.pct_change()

daily_pnl = daily_pnl['2015-04-16':]
NAV = (daily_pnl.T.sum()+1).cumprod() #计算净值
NAV0 = 1+(daily_pnl.T.sum()).cumsum() #计算净值

IC = IC.resample('M').last()
IF = IF.resample('M').last()

#画图
plt.figure(figsize=(8,6))
NAV.plot(label='Selection')
if 大盘股:
    (hs300.pct_change()+1).cumprod().plot(label='000300.SH')
    (NAV/(hs300.pct_change()+1).cumprod()).plot(label='Exess Return')
else:
    (IC.CLOSE.pct_change()+1).cumprod().plot(label='000905.SH')
    (NAV/(IC.CLOSE.pct_change()+1).cumprod()).plot(label='Exess Return')
plt.legend(fontsize=14)
