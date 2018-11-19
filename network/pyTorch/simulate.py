# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:34:06 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import sys
local_path = 'C://Users//Administrator/Desktop/'
if local_path not in sys.path:
    sys.path.append(local_path)
from get_data import data_reader as rd

def maxback(x):
    
    local_max = 1 
    back = 0
    
    for i in range(len(x)):
        if x[i]> local_max:
            local_max = x[i]
        if (local_max - x[i])/local_max > back:
            back = (local_max - x[i])/local_max
    return back



def run(stock, N=4):
    
    data = rd.get_index_day(stock,'2010-01-01','2018-07-18','1D')
    signal = [0]
    val=[1]
    for i in range(30,len(data)-1):
        Return = (data.sclose.iloc[i+1])/(data.sclose.iloc[i])
        yesterday_return = (data.sclose.iloc[i])/(data.sclose.iloc[i-1])
        vol = data.volumn.iloc[i-30:i].values
        cnt = 0
        for j in vol:
            if vol[-1] <= j:
                cnt = cnt+1
        if cnt <= N:
            signal.append(1)
            val.append(val[-1]+Return-1)
        elif yesterday_return >2:
            signal.append(1)
            val.append(val[-1]+Return-1)
        else:
            signal.append(0)
            val.append(val[-1])
        
        if signal[-1] !=signal[-2]:
            
            val[-1] = val[-1]*0.999
    return val,signal

def evalue(stock):
    data = rd.get_index_day(stock,'2010-01-01','2018-07-18','1D')
    
    benchmark = data.sclose.iloc[30:].values.tolist()
    
    a,b = run(stock)
    
    df = pd.DataFrame(data = (np.array([benchmark,a]).T),index = data.index[30:],columns = [stock,'strategy'])
    
    df = df/df.iloc[0]
    
    df.plot()
    
    print(maxback(a))
    
    return df,b
    