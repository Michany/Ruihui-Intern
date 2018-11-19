# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 13:58:11 2018

@author: Administrator
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import talib as ta
import sys
local_path = 'C://Users//Administrator/Desktop/'
if local_path not in sys.path:
    sys.path.append(local_path)
from get_data import data_reader as rd
from get_data import tic_toc

rf = RandomForestClassifier()



SS = rd.get_index_day('000905.SH','2000-01-01','2018-07-01',freq = '1D')

target = []
pct = SS.sclose.pct_change().dropna()
v = pct.values

for i in v:
    
    if i>0.01:
        target.append(3)
        
    elif i<-0.01:
        
        target.append(0)
        
    elif i >0.5:
        
        target.append(2)
        
    elif i<-0.05:
        target.append(1)
    else:
        target.append(5)
        
        
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

train = ta_index[:'2010-01-01']


x = train.values[:,-1]

input = train.values[:,4:-1]

rf = RandomForestClassifier(n_estimators=100)

predict = ta_index['2010-01-01':]
close = predict.iloc[:,:4]
predict = predict.iloc[:,4:-1]

rf.fit(input,x)

val =[1]

for i in range(len(predict)-1):
    parameters = predict.values[i]
    x = rf.predict([parameters])[0]
    if int(x) == 0 :
        val.append(0.999*val[-1]*(2-close.sclose.iloc[i+1]/close.sclose.iloc[i]))
    elif int(x) == 3:
        val.append(0.999*val[-1]*close.sclose.iloc[i+1]/close.sclose.iloc[i])
    elif int(x) == 1:
        val.append(0.5*0.999*val[-1]*close.sclose.iloc[i+1]/close.sclose.iloc[i]+0.5*val[-1])
    elif int(x) == 2:
        val.append(0.5*0.999*val[-1]*(2-close.sclose.iloc[i+1]/close.sclose.iloc[i])+0.5*val[-1])
    else:
        val.append(val[-1])
    
df = pd.DataFrame(index = predict.index,data=val)
df.plot()