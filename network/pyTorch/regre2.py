# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 21:04:20 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
read_path = r'C:\Users\Administrator\Desktop'
if read_path not in sys.path:
    sys.path.append(read_path)

from get_data import data_reader as rd




today = datetime.datetime.today()
today = str(today)[:10]


ss_min = rd.get_index_min('000300.SH','2010-01-01',today,'5min')

close = ss_min.sclose
Return = close.pct_change()
Return = pd.DataFrame(Return)
"""
40 day = 320 30min = 1920 5min
"""
D = 30
N = D*48
n = 48
def simple_regression(x):
    
    x = np.array(x)
    
    beta,alpha = np.polyfit(np.arange(len(x)),x,deg=1)
    
    return beta

day_close = close.resample('1D').last().dropna()
day_close = pd.DataFrame(day_close)
day_close['beta'] = day_close.rolling(D).apply(simple_regression,raw=True)

STD = day_close.beta.rolling(D).std()

def regression(data,tail):
    y = data.values
    x = np.arange(len(y))
    beta,alpha = np.polyfit(x,y,deg=1)
    res = y - x*beta - alpha
    R = res.var() / y.var()
    res_tail = res[-tail:]
    dev = res_tail.var()/ res.var()
    return beta,dev


def regre_strat(factor=1):
    signal =[]
    val = [1]
    k=0
    beta_list=[]
    work=[]
    for i in range(2*(D-1)*48,len(Return)-n,n):
        std = STD[k]
        day_close = close.iloc[i-N:i].resample('1D').last().dropna()
        day_beta,day_dev = regression(day_close,4)
        beta = day_beta
        yday_return = close.iloc[i]/close.iloc[i-48]
        day_return = close.iloc[i+48]/close.iloc[i]
        k = k+1
        work.append(1)
        if day_dev>1:
            min30_close = close.iloc[i-4*40:i].resample('30min').last().dropna()
            min30_beta,min30_dev = regression(min30_close,8)
            beta = min30_beta*8
            work[-1]=2
            if min30_dev >1:
                min5_close = close.iloc[i-6*8:i].resample('5min').last().dropna()
                min5_beta,min5_dev = regression(min5_close,8)
                beta = min5_beta*48
                work[-1]=3
        beta_list.append(beta)
        if abs(beta) < abs(std)*factor:
            val.append(val[-1])
            signal.append(0)

        else:
            if beta >0:
                val.append(val[-1]*day_return)
                signal.append(1)
            else:
                val.append(val[-1]*(2-day_return))
                signal.append(-1)
    return val,signal,beta_list,work


factor_cluster = np.linspace(1,2,10)
Return_list =[]
signal_list =[]

def calc():
    for i in factor_cluster:
    
    
    
        val,signal = regre_strat(i)
        Return_list.append(val)
        signal_list.append(signal)
    return Return_list,signal

Return_list,signal = calc()

def crit(start='2015-01-01',end='2018-07-01'):
    
    for i,j in enumerate(Return_list):
        fig = plt.figure()
        print('')
        j = pd.Series(j)
        j = j.iloc[1:]
        victory = j[j.pct_change()>0].count()/(j[j.pct_change()!=0].count())
        hold = j[j.pct_change()!=0].count()/j.count()
        print(factor_cluster[i])
        print("Return",j.iloc[-1])
        print("VIC",victory)
        print("Hold",hold)
        j = j.values
        data = pd.DataFrame(data =np.array([j,signal_list[i],ss.sclose.iloc[81:]]).T,index = ss.index[81:],columns=['val','signal','SS'])
        data = data[start:end]
        data.SS = data.SS/data.SS[0]*data.val[0]
        ax1 = fig.add_subplot(111)
        ax1.plot(data.index,data.val,label=str(i))
        ax1.plot(data.index,data.SS,label='SS')
        ax2 = ax1.twinx()
        ax2.scatter(data.index,data.signal,alpha=0.2,s=5,color='r',label='holding')
        plt.show()