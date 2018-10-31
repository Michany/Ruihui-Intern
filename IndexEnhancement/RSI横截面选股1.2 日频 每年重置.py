# -*- coding: utf-8 -*-
"""
@author: Michael Wang

`Based on Version 1.1`

- 港股
- 市值大于100亿
- 5e7
- 每年重置资本金
- RSI线 分为快慢
updated on 2018/10/31
"""
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from data_reader import get_muti_close_day, get_index_day, get_hk_index_day
import pymssql
# pylint: disable=E1101,E1103
# pylint: disable=W0212,W0231,W0703,W0622
CAPITAL = 5e7

# 获取标的数据
underLying = 'hs300'#zz500
if underLying == 'hs300':
    conn = pymssql.connect(server='192.168.0.28', port=1433, user='sa', password='abc123', database='WIND')
    SQL = '''SELECT b.code FROM HS300COMPWEIGHT as b
    where b.[Date] BETWEEN '2018-07-01' and '2018-07-03' ORDER BY b.code'''
    hs300 = get_index_day('000300.SH', '2007-02-01', datetime.date.today().strftime('%Y-%m-%d'), '1D')
    hs300 = hs300.sclose
elif underLying == 'zz500':
    conn = pymssql.connect(server='192.168.0.28', port=1433, user='sa', password='abc123', database='RawData')
    SQL = '''SELECT b.code FROM [中证500成分权重] as b
    where b.[Date] BETWEEN '2018-06-21' and '2018-07-03' '''
    zz500 = get_index_day('000905.SH', '2007-02-01', datetime.date.today().strftime('%Y-%m-%d'), '1D')
    zz500 = zz500.sclose
elif underLying == 'hsi':
    hsi = get_hk_index_day('HSI.HI', '2007-02-01', datetime.date.today().strftime('%Y-%m-%d'), '1D')
    hsi = hsi.sclose
elif underLying == 'zz800':
    conn1 = pymssql.connect(server='192.168.0.28', port=1433, user='sa', password='abc123', database='WIND')
    SQL1 = '''SELECT b.code FROM HS300COMPWEIGHT as b where b.[Date] BETWEEN '2018-07-01' and '2018-07-03' ORDER BY b.code'''
    conn2 = pymssql.connect(server='192.168.0.28', port=1433, user='sa', password='abc123', database='RawData')
    SQL2 = '''SELECT b.code FROM [中证500成分权重] as b where b.[Date] BETWEEN '2018-06-21' and '2018-07-03' '''
    zz800 =  get_index_day('000906.SH', '2007-02-01', datetime.date.today().strftime('%Y-%m-%d'), '1D')
    zz800 = zz800.sclose
if underLying=='hs300' or underLying=='zz500':
    data = pd.read_sql(SQL, conn)
elif underLying == 'zz800':
    data1 = pd.read_sql(SQL1, conn1)
    data2 = pd.read_sql(SQL2, conn2)
    data = pd.concat([data1, data2], ignore_index=True)
    del data1, data2
elif underLying=='hsi':
    data = pd.read_excel(r"C:\Users\meiconte\Documents\RH\IndexEnhancement\HK100亿_01.xlsx",sheet_name='Database')
pool = list(data['code'])
del data

# 
def 获取数据():
    global price
    START_DATE = '2007-02-01'
    END_DATE = datetime.date.today().strftime('%Y-%m-%d')
    print('正在获取数据...自 {} 至 {}'.format(START_DATE, END_DATE))
    price = get_muti_close_day(pool, START_DATE, END_DATE, HK=(underLying=='hsi'))
    price.fillna(method="ffill", inplace=True)
    print("Historical Data Loaded!")
获取数据()

#%% 
def 仓位计算和优化(arg=30, fast = False):
    global RSI_arg, RSI

    RSI_arg = arg
    RSI = price.apply(getattr(talib, 'RSI'), args=(RSI_arg,)).shift(1)
    RSI=RSI.replace(0,np.nan)
#    if fast:
#        RSI = 100-RSI
    分母=abs(RSI.T-50).sum()
    RSI_normalized = ((RSI.T-50)/分母).T
    RSI_normalized.fillna(0,inplace=True)
    pos = RSI_normalized[RSI_normalized>0]

    # rsi一快一慢 搭配使用？
    # 当 慢pos的仓位很低时，使用快pos作为信号
    # 快的pos可能需要不同的处理
    # 在rsi都往上的时候，有可能rsi阈值需要提高到50以上，比如65，然后通过少选股票，增加仓位来做
    # pos[pos.T.sum()>0.1] *= 1.5
    # pos[pos.T.sum()>0.2] *= 1.2
    # pos[pos.T.sum()>0.3] *= 1.2
    pos[pos.T.sum()<0.4] *= 0.8
    pos[pos.T.sum()<0.3] *= 0.5
#if not fast:
    pos[pos.T.sum()>0.6] *= 1.1
    pos[pos.T.sum()>0.7] *= 1.1
    pos[pos.T.sum()>0.8] *= 1.1
    # 将总和超出1的仓位，除以总和，归为1
    pos[pos.T.sum()>1] = pos[pos.T.sum()>1].divide(pos.T.sum()[pos.T.sum()>1],axis=0)
    pos.fillna(0, inplace = True)

    return pos
posOriginal = 仓位计算和优化(30)
posSlow = 仓位计算和优化(30)
posFast = 仓位计算和优化(5, fast=True)
posSlow[(posSlow.T.sum()<0.50) & (posSlow.T.sum()>0.05)] = posFast
posSlow[(posSlow.T.sum()>0.99) & (posFast.T.sum()<0.32)] = posFast

def check(y):#看一下具体每一年的表现
    year = str(y)
    posSlow[year].T.sum().plot(c='r')
    plt.twinx()
    hs300[year].plot()
    plt.show()
    
    posFast[year].T.sum().plot(c='r')
    plt.twinx()
    hs300[year].plot()
    plt.show()
    
    NAV0[year].plot(c='r')
    (hs300[year]/hs300[year].iloc[0]).plot()
    plt.show()
#%%
# share记录了实时的仓位信息
# 注意：交易时间为交易日的收盘前
share = posSlow#[pos.index.dayofweek == 交易日]
share = share.reindex(posSlow.index)
share.fillna(method='ffill',inplace=True)
price_pct_change = price.pct_change().replace(np.inf,0)

#近似 & 按月复利 按年重置
daily_pnl=pd.DataFrame()
#capitalBalance = pd.Series()
for year in range(2008,2019):
    initialCaptial = CAPITAL
    for month in range(1,13):
        this_month = str(year)+'-'+str(month)
        try:
            temp = round(share[this_month] * initialCaptial/ price,-2)
        except:
            continue
        share[this_month] = temp.fillna(method='ffill')
        daily_pnl = daily_pnl.append(price_pct_change[this_month] * (share[this_month]*price[this_month]))
        initialCaptial += daily_pnl[this_month].T.sum().sum()
#        print(this_month, initialCaptial)
# 手续费，卖出时一次性收取
fee_rate = 0.0013
fee = (share.diff()[share<share.shift(1)] * price * fee_rate).fillna(0).abs()
daily_pnl -= fee
# 按年清空
cum_pnl = pd.DataFrame(daily_pnl.T.sum(),columns=['pnl'])
cum_pnl['year']=cum_pnl.index.year
cum_pnl = cum_pnl.groupby('year')['pnl'].cumsum()
NAV = (daily_pnl.T.sum()/CAPITAL).cumsum()+1
NAV0 = (cum_pnl / CAPITAL)+1
#换手率
换手率=((share * price).divide((share * price).T.sum(),axis=0).diff().abs().T.sum() / 2)
print("每日换手率 {:.2%}".format(换手率.mean()))
print("年化换手率 {:.2%}".format(换手率.mean()*250))
print(换手率.resample('y').sum())

def 图像绘制():
    global hs300
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(9,6))
    
    NAV.plot(label='按月复利 每年重置 累计值')
    NAV0.plot(label='按月复利 每年重置')
    exec(underLying+' = '+underLying+".reindex(daily_pnl.index)")
    exec(underLying+' = '+underLying+'/'+underLying+'.iloc[0]')
    exec(underLying+".plot(label='"+underLying+"')")
    plt.legend(fontsize=11)
    # plt.title('RSI参数={}，日频，无手续费'.format(RSI_arg),fontsize=15)
    plt.title('RSI参数={}，日频，手续费{:.1f}‰'.format(RSI_arg, fee_rate*1000), fontsize=15)
    # plt.title('RSI参数={}，交易日={}，手续费{:.1f}‰'.format(RSI_arg, 交易日+1, fee_rate*1000), fontsize=15)
    plt.grid(axis='x')
    plt.show()
图像绘制()
#%% 
def excel输出():
    df = pd.DataFrame({'Daily_pnl':daily_pnl.T.sum(),
                       '累计PNL':cum_pnl,
                       '账户价值':cum_pnl+CAPITAL,
                       'NAV':NAV0, 'NAV累计':NAV},
                       index = daily_pnl.index)
    df.to_excel('RSI横截面_{}纯多头_收益率明细_{}_日.xlsx'.format(underLying, datetime.date.today().strftime('%y-%m-%d')),
                sheet_name = 'RSI={},日频'.format(RSI_arg))
    share.to_excel('RSI横截面_{}纯多头_持仓明细_{}_日.xlsx'.format(underLying,datetime.date.today().strftime('%y-%m-%d')),
                sheet_name = 'RSI={},日频'.format(RSI_arg))
excel输出()
print('2017年收益：{:.2%}'.format(daily_pnl['2017'].T.sum().sum()))
#%% 敏感性
# exec('daily_pnl_'+str(RSI_arg)+'_交易日'+str(交易日+1)+"=daily_pnl")
# exec('NAV_'+str(RSI_arg)+'_交易日'+str(交易日+1)+"=NAV")
# # RSI参数
# plt.figure(figsize=(15,9))
# for arg in range(15,31,5):
#     exec('NAV_'+str(arg)+"_交易日4.plot(label='RSI="+str(arg)+"_交易日4"+"')")
# plt.legend(fontsize=12)
# # 交易日
# plt.figure(figsize=(15,9))
# for day in range(1,6):
#     exec('NAV_30'+"_交易日"+str(day)+".plot(label='RSI=30"+"_交易日"+str(day)+"')")
# plt.legend(fontsize=12)