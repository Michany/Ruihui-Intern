# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 09:31:38 2018

@author: Administrator
"""


import numpy as np
import pandas as pd
import sys
local_path = 'C://Users//Administrator/Desktop/'
if local_path not in sys.path:
    sys.path.append(local_path)
from get_data import data_reader as rd



weight = pd.read_excel('C:/Users/Administrator/Desktop/成份及权重.xlsx')






def run(index, N=4):
    
    data = rd.get_index_day(index,'2007-01-15','2018-05-01')
    val =[1]
    signal =[]
    
    for i in range(30,len(data)-1):
        amt = data.amt.iloc[i-30:i].values
        cnt = 0
        for j in amt:
            if amt[-1] <= j:
                cnt = cnt+1
        if cnt <= N:
            signal.append(1)
            Return = (data.sclose.iloc[i+1])/(data.sclose.iloc[i])
            val.append(val[-1]*Return)
        else:
            signal.append(0)
            val.append(val[-1])
    return val,signal

def evalue(stock):
    
    data = rd.get_index_day(stock,'2007-01-15','2018-05-01')
    
    benchmark = data.sclose.iloc[30:].values.tolist()
    
    a,b = run(stock)
    
    df = pd.DataFrame(data = (np.array([benchmark,a]).T),index = data.index[30:],columns = [stock,'strategy'])
    
    df = df/df.iloc[0]
    
    df.plot()