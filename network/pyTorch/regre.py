# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:11:35 2018

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

reg = linear_model.LinearRegression()


today = datetime.datetime.today()
today = str(today)[:10]

ss = rd.get_index_day('000001.SH','2010-01-01',today,'1D')
ss_min = rd.get_index_min('000300.SH','2010-01-01',today,'5min')

close = ss.sclose
Return = close.pct_change()
Return = pd.DataFrame(Return)

def regression(x):
    
    x = np.array(x)
    
    beta,alpha = np.polyfit(np.arange(len(x)),x,deg=1)
    
    return beta

N = 40

Return['beta'] = Return.rolling(N).apply(regression,raw=True)
Return = Return.dropna()
std = Return.beta.rolling(N).std()


factor = 1

def regre_strat(factor=1):
    signal =[]
    val = [1]
    for i in range(N,len(Return)-1):
        if abs(Return['beta'].iloc[i]) < std[i-N]*factor:
            val.append(val[-1])
            signal.append(0)
        else:
            if Return.sclose.iloc[i]>0:
                day_return = 1+Return.sclose.iloc[i+1]
                val.append(val[-1]*day_return)
                signal.append(1)
            else:
                day_return = 1-Return.sclose.iloc[i+1]
                val.append(val[-1]*day_return)
                signal.append(-1)
    return val,signal

def regre_strat_NOSHORT(factor=1):
    signal =[]
    val = [1]
    for i in range(N,len(Return)-1):
        if abs(Return['beta'].iloc[i]) < std[i-N]*factor:
            val.append(val[-1])
            signal.append(0)
        else:
            if Return.sclose.iloc[i]>0:
                day_return = 1+Return.sclose.iloc[i+1]
                val.append(val[-1]*day_return)
                signal.append(1)
            else:
                 val.append(val[-1])
            signal.append(0)
    return val,signal

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