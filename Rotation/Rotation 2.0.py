# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:59:38 2018

@author: Michael Wang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib as mpl

import pymssql
def get_findata(SQL,write_to_file = 'off'):
    conn=pymssql.connect(server='192.168.0.28', port=1433, user='sa', password='abc123', database='rawdata',charset = 'GBK')
    data = pd.read_sql(SQL,conn)
    if write_to_file == 'on':
        data.to_csv('data.csv') 
    return data

PE = get_findata("select * from dbo.Zindexes_PE")
pe = PE.dropna(thresh = 30).set_index('Date')
# pe从日频改成周频
pe = pe.resample('1W').last()
indexes = list(pe.columns)
#%% 获取收盘数据
from data_reader import get_index_day, get_HKindex_min
start_date = str(pe.index[0])[:10]
end_date = str(pe.index[-1])[:10]

close = dict();symbols=[]
for index in indexes:
    symbol = index[index.find("TTM_")+4:]
    symbol = symbol.replace('_','.')
    if symbol in ['WDQA','884153.WI']:
        temp = pd.read_excel(r"E:\RH\指数行情序列.xlsx", dtype={'date':'datetime64'})
        temp.set_index('date', inplace=True)
        temp = temp[symbol].resample('1W').first()
        close[index] = temp
        symbols.append(symbol)
        print(symbol, 'success!')
        continue
    elif symbol == 'SZ50':
        symbol = '000016.SH'
    elif symbol == 'SH':
        symbol = '000001.SH'
    elif symbol == 'ZZ500':
        symbol = '000905.SH'
    elif symbol == 'CYB':
        symbol = '399006.SZ'
    elif symbol == "HS300":
        symbol = '399300.sz'
    elif symbol in ['HSI.HI','HSCEI.HI']:
        symbol = symbol.split('.')[0]
        temp = get_HKindex_min(symbol, start_date, end_date, freq = '1W')
        close[index] = temp.sclose
        symbols.append(symbol)
        continue
    try:
        temp = get_index_day(symbol, start_date, end_date, freq = '1W')
        close[index] = temp.sclose
        print(symbol, 'success!')
    except:
        print(symbol, "failed")
    symbols.append(symbol)
close = pd.DataFrame(close,index=pe.index)


#%% 计算移动平均线,相对PE
rpe={};mean={}
for index in indexes: 
    #计算移动平均线
    mean[index]=close[index].rolling(5).mean()
    #计算相对PE
    temp=pe[index]/pe[index][1]
    rpe[index] = temp
rpe=pd.DataFrame(rpe)
mean=pd.DataFrame(mean)
#for index in indexes: 
#    rpe.loc[26:,index]=pe[index]/pe[index].shift(26)
    
#%% 计算仓位
index_amount = 3
frequency = 3
rank=rpe.rank(axis=1)
# 调仓时间点
chg_time = pd.Series(np.arange(len(pe)) % frequency, index = pe.index)
chg_pos =  chg_time == 0    
#赋权重(仓位)
pos = rpe.copy()
longer_pos = rpe.copy()
for index in indexes:
    pos[index] = rank[index].apply(lambda x: 1/index_amount if x<=index_amount else 0) 
    for i in range(len(pos.index)):
        if chg_pos.iloc[i]:
            longer_pos.loc[pos.index[i],index] = pos[index].iloc[i]
        else:
            longer_pos.loc[pos.index[i],index] = pos[index].iloc[i-chg_time.iloc[i]]
#设置止损
#如果该市场指数下跌跌破均线，则止损
for index in indexes:
    pos.loc[(close[index]<mean[index]) & (close[index]<close[index].shift(2)), index] = 0
#如果市场指数上穿均线，则开仓
    
          
#%% 计算收益
profit, profit_l = pe, pe
cum_profit={}
for index in indexes:
    profit[index] = (close[index].shift(-1)/close[index] - 1) * pos[index]

cum_profit['single'] = 1 + (profit.T.sum()).cumsum() 
cum_profit['compound'] = (profit.T.sum()+1).cumprod() 
cum_profit=pd.DataFrame(cum_profit, index=pe.index)

#计算收益 3周调仓1次
for index in indexes:
    profit_l[index] = (close[index].shift(-1)/close[index] - 1) * longer_pos[index]

cum_profit['single_longer'] = 1 + (profit_l.T.sum()).cumsum() 
cum_profit['compound_longer'] = (profit_l.T.sum()+1).cumprod() 


#%% 画图
cum_profit['single'].plot(c='gold',alpha=0.9,figsize=(12,10),linewidth=3)
#cum_profit['compound'].plot(c='orange',alpha=0.9,linewidth=3)
cum_profit['single_longer'].plot(c='red',alpha=0.9,linewidth=3)
#cum_profit['compound_longer'].plot(c='brown',alpha=0.9,linewidth=3)
'''    
relative_index = pd.DataFrame((close[index]/close[index][0]) for index in indexes).T
for index in indexes:
    plt.plot(relative_index[index],alpha=0.7,label=index)   
'''

def generate_profit_curve(ans: pd.DataFrame, column='single'):
    fig = plt.figure()
    fig.set_size_inches(18, 12)
    ans['NAV'] = ans[column]; ans['Daily_pnl'] = ans[column].diff()
 
    ax = fig.add_subplot(211)
    ax.plot(ans.index, ans['NAV'], linewidth=2, label='净值')
    ax.fill_between(ans.index, ans['NAV'], y2=1,
                    where=(ans['NAV'] < ans['NAV'].shift(1)) |
                    ((ans['NAV'] > ans['NAV'].shift(-1)) &
                     (ans['NAV'] >= ans['NAV'].shift(1))),
                    facecolor='grey',
                    alpha=0.3)
    # 最大回撤标注
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax.legend(fontsize=15)
    plt.grid()

    bx = fig.add_subplot(212)
    width = 1
    if len(ans)>30: width = 1.5
    bx.bar(ans.index, ans['Daily_pnl'].where(ans['Daily_pnl'] > 0),
           width, label='当日盈亏+', color='red', alpha=0.8)
    bx.bar(ans.index, ans['Daily_pnl'].where(ans['Daily_pnl'] < 0),
           width, label='当日盈亏-', color='green', alpha=0.8)
    bx.legend(fontsize=15)
    plt.grid()
generate_profit_curve(cum_profit)