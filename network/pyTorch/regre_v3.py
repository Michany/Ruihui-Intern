# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 08:27:36 2018

@author: Administrator
"""

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

def calc_dev(data,history_data,tail,history_beta,history_alpha):
    y = data.values
    x = np.arange(len(data))
    dy = history_data.values
    dx = np.arange(len(dy))
    dres = dy-dx*history_beta - history_alpha 
    beta,alpha = np.polyfit(x,y,deg=1)
    res = y[-tail:] - np.arange(len(data),len(data)+tail)*history_beta - history_alpha
    dev = res.var()/ dres.var()
    return dev,beta

def history_regression(data,history =True):
    y = data.values
    x = np.arange(len(y))
    beta,alpha = np.polyfit(x,y,deg=1)
    return beta,alpha


def regre_strat(beta_limit=1,dev_limit = 1,day_tail=5,min30_tail=8):
    signal =[]
    val = [1]
    k=0
    beta_list=[]
    work=[]
    n = 48
    for i in range(2*(D-1)*48,len(Return)-n,n):
        std = STD[k]
        day_close = close.iloc[i-N:i].resample('1D').last().dropna()
        history_day_close = close.iloc[i-N-48*day_tail:i-48*day_tail].resample('1D').last().dropna()
        history_day_beta,history_day_alpha = history_regression(history_day_close)
        day_dev,day_beta = calc_dev(day_close,history_day_close,day_tail,history_day_beta,history_day_alpha)   #一天 48 5min 8 30min
        beta = day_beta
        day_return = close.iloc[i+48+47]/close.iloc[i+47]
        k = k+1
        work.append(1)
        if day_dev>dev_limit:
            min30_close = close.iloc[i-day_tail*48:i].resample('30min').last().dropna()
            history_min30_close = close.iloc[i-day_tail*48-min30_tail*6:i-min30_tail*6].resample('30min').last().dropna()
            history_min30_beta,history_min30_alpha = history_regression(history_min30_close)
            min30_dev,min30_beta = calc_dev(min30_close,history_min30_close,min30_tail,history_min30_beta,history_min30_alpha)
            beta = min30_beta*8
            work[-1]=2
            if min30_dev >dev_limit:
                min5_close = close.iloc[i-min30_tail*6:i].resample('5min').last().dropna()
                min5_beta,___= history_regression(min5_close,history = False)
                beta = min5_beta*48
                work[-1]=3
        beta_list.append(beta)
        if abs(beta) < beta_limit:
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



Return_list =[]
signal_list =[]
val,_,__,___ = regre_strat(beta_limit=10,dev_limit=1.5,day_tail=5,min30_tail=5)
df = pd.DataFrame(data = np.array([val,day_close.sclose.iloc[2*(D-1):]]).T,index = day_close.index[2*(D-1):],columns = ['strategy','benchmark'])


'''
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
'''