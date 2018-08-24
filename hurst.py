from __future__ import division
from collections import Iterable

import numpy as np 
import pandas as pd
from pandas import Series

def hurst(history):
    daily_return = list(history.pct_change())[1:]
    ranges = ['1','2','4','8','16','32']
    lag = Series(index = ranges)
    for i in range(len(ranges)):
        if i==0:
            lag[i] = len(daily_return)
        else:
            lag[i] = lag[0]//(2**i)

    ARS = Series(index = ranges)
    
    for r in ranges:
        #RS用来存储每一种分割方式中各个片段的R/S值
        RS = list()
        #第i个片段
        for i in range(int(r)):
            #用Range存储每一个片段数据
            Range = daily_return[int(i*lag[r]):int((i+1)*lag[r])]
            mean = np.mean(Range)
            Deviation = Range - mean
            maxi = max(Deviation)
            mini = min(Deviation)
            RS.append(maxi - mini)
            sigma = np.std(Range)
            RS[i] = RS[i]/sigma
        
        ARS[r] = np.mean(RS)
    
    lag = np.log10(lag)
    ARS = np.log10(ARS)
    hurst_exponent = np.polyfit(lag,ARS,1)
    hurst = hurst_exponent[0]*2
    return hurst

t=pd.read_csv(r"E:\WinPython-32bit-3.5.3.0Qt5\Data2\AAPL.csv")
for i in range(1,100):
    print(hurst(t['Close'].loc[i:i+100]))