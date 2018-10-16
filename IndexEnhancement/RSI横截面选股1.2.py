# -*- coding: utf-8 -*-
"""
@author: Michael Wang

`Based on Version 1.1`

- 按月复利！

updated on 2018/10/16
"""
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from data_reader import get_muti_close_day, get_index_day
import pymssql

# 获取标的数据
underLying = 'hs300'#zz500
if underLying=='hs300':
    conn=pymssql.connect(server='192.168.0.28',port=1433,user='sa',password='abc123',
                         database='WIND') 
    SQL='''SELECT b.code FROM HS300COMPWEIGHT as b
    where b.[Date] BETWEEN '2018-07-01' and '2018-07-03' ORDER BY b.code'''
    hs300 = get_index_day('000300.SH','2007-02-01',datetime.date.today().strftime('%Y-%m-%d'),'1D')
    hs300 = hs300.sclose

elif underLying=='zz500':
    conn=pymssql.connect(server='192.168.0.28',port=1433,user='sa',password='abc123',
                         database='RawData') 
    SQL='''SELECT b.code FROM [中证500成分权重] as b
    where b.[Date] BETWEEN '2018-06-21' and '2018-07-03' '''
    zz500 = get_index_day('000905.SH','2007-02-01',datetime.date.today().strftime('%Y-%m-%d'),'1D')
    zz500 = zz500.sclose   


data = pd.read_sql(SQL,conn)
pool = list(data['code'])
del data

# 
def 获取数据():
    global price
    START_DATE = '2007-02-01'
    END_DATE = datetime.date.today().strftime('%Y-%m-%d')
    print('正在获取数据...自 {} 至 {}'.format(START_DATE, END_DATE))
    price = get_muti_close_day(pool, START_DATE, END_DATE)
    price.fillna(method="ffill", inplace=True)
    print("Historical Data Loaded!")
获取数据()

#%% 
def 仓位计算和优化():
    global pos,RSI_arg

    RSI_arg = 30
    RSI = price.apply(getattr(talib, 'RSI'), args=(RSI_arg,))
    RSI=RSI.replace(0,np.nan)
    分母=abs(RSI.T-50).sum()
    RSI_normalized = ((RSI.T-50)/分母).T
    RSI_normalized.fillna(0,inplace=True)
    pos = RSI_normalized[RSI_normalized>0]
    #pos = pos.multiply((RSI_normalized.T.sum().T + 0.3),axis=0)
    #pos[pos<0] = 0
    #pos[pos.T.sum()>1] /= pos.T.sum()
    pos[pos.T.sum()>0.6] *= 1.1
    pos[pos.T.sum()>0.7] *= 1.1
    pos[pos.T.sum()>0.8] *= 1.1
    pos[pos.T.sum()<0.4] *= 0.8
    pos[pos.T.sum()<0.3] *= 0.5
    # 将总和超出1的仓位，除以总和，归为1
    pos[pos.T.sum()>1] = pos[pos.T.sum()>1].divide(pos.T.sum()[pos.T.sum()>1],axis=0)
    pos.fillna(0, inplace = True)
仓位计算和优化()

# share记录了实时的仓位信息
# 注意：交易时间为交易日的收盘前
交易日=4
share = pos[pos.index.dayofweek == 交易日]
share = share.reindex(pos.index)
share.fillna(method='ffill',inplace=True)
price_pct_change = price.pct_change().replace(np.inf,0)

#近似
daily_pnl=pd.DataFrame()
NAV = pd.Series()
initialCaptial = 1e6
for year in range(2008,2019):
    for month in range(1,13):
        this_month = str(year)+'-'+str(month)
        try:
            temp = round(share[this_month] * initialCaptial/ price,-2)
        except:
            continue
        share[this_month] = temp.fillna(method='ffill')
        daily_pnl = daily_pnl.append(price_pct_change[this_month] * (share[this_month]*price[this_month]).shift(1))
        initialCaptial += daily_pnl[this_month].T.sum().sum()
        NAV = NAV.append((daily_pnl[this_month].T.sum() / 1e6))
        print(this_month, initialCaptial)

# 手续费，卖出时一次性收取
fee_rate = 0.0013
fee = (share.diff()[share<share.shift(1)] * price * fee_rate).fillna(0).abs()
daily_pnl -= fee
NAV = NAV.cumsum()+1



def 图像绘制():
    global hs300
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(9,6))

    NAV.plot(label='单利')
    hs300 = hs300.reindex(daily_pnl.index)
    hs300 /= hs300.iloc[0]
    hs300.plot(label='HS300')
    plt.legend(fontsize=15)
    # plt.title('RSI参数={}，日频，无手续费'.format(RSI_arg),fontsize=15)
    plt.title('RSI参数={}，日频，手续费{:.1f}‰'.format(RSI_arg, fee_rate*1000), fontsize=15)
    plt.show()
图像绘制()
#%% 
def excel输出():
    df = pd.DataFrame({'Daily_pnl':daily_pnl.T.sum(), 'NAV':NAV},index = daily_pnl.index)
    df.to_excel('RSI横截面_纯多头_收益率明细_{}.xlsx'.format(datetime.date.today().strftime('%y-%m-%d')),
                sheet_name = 'RSI={},周{},周频'.format(RSI_arg,交易日+1))
    share.to_excel('RSI横截面_纯多头_持仓明细_{}.xlsx'.format(datetime.date.today().strftime('%y-%m-%d')),
                sheet_name = 'RSI={},周{},周频'.format(RSI_arg,交易日+1))
excel输出()
print('2017年收益：{:.2%}'.format(daily_pnl['2017'].T.sum().sum()))

