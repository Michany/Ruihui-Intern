# -*- coding: utf-8 -*-
"""
@author: Michael Wang

`Based on Version 1.1`

- 港股
- 市值大于100亿
- 初始资本金 5e7
- 每年重置资本金
- RSI线 分为快慢
- price 分为交易/停牌，需要区分对待；对于停牌的，需要用nan，并在计算中不予以考虑
- 模拟盘PnL监测
- Wind数据 10个一组 获取
updated on 2019/1/17
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
YEAR = 2012
START = '%s-11-01' % (YEAR-1) #前三个月的数据需要用来计算RSI
TODAY = '%s-12-31' % YEAR#datetime.date.today().strftime('%Y-%m-%d')

# 获取标的数据
underLying = 'hs300'#zz500
if underLying == 'hs300':
    conn = pymssql.connect(server='10.0.0.51', port=1433, user='sa', password='abc123', database='WIND')
    SQL = '''SELECT b.code FROM HS300COMPWEIGHT as b
    where b.[Date] BETWEEN '%s-07-01' and '%s-07-30' ORDER BY b.code''' % (YEAR, YEAR)
    hs300 = get_index_day('000300.SH', START, TODAY, '1D')
    hs300 = hs300[str(YEAR)].sclose

if underLying=='hs300' or underLying=='zz500':
    data = pd.read_sql(SQL, conn)
pool = list(data['code'])
del data

# 
def 获取数据():
    START_DATE = START
    END_DATE = TODAY
    print('正在获取数据...自 {} 至 {}'.format(START_DATE, END_DATE))
    price = get_muti_close_day(pool, START_DATE, END_DATE, HK=(underLying=='hsi'))
    priceFill = price.fillna(method="ffill")
    print("Historical Data Loaded!")
    return price, priceFill
price, priceFill = 获取数据()

#%% 
def 仓位计算和优化(arg=30, fast = False):
    global RSI_arg

    RSI_arg = arg
    RSI = priceFill.apply(talib.RSI, args=(RSI_arg,))

    RSI[price.isna()] = 50

    分母=abs(RSI.T-50).sum()
    RSI_normalized = ((RSI.T-50)/分母).T
    RSI_normalized.fillna(0,inplace=True)
    pos = RSI_normalized[RSI_normalized>0]

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

    return pos, RSI
posOriginal = 仓位计算和优化(40)
posSlow, RSI_Slow = 仓位计算和优化(40)
posFast, RSI_Fast = 仓位计算和优化(10, fast=True)
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
# 交易时间为交易日的收盘前
share = posSlow#[pos.index.dayofweek == 交易日]
share = share.reindex(posSlow.index)
share.fillna(method='ffill',inplace=True)
price_change = priceFill.diff()

# 近似 & 按月复利 & 按年重置
daily_pnl=pd.DataFrame()
share_last_month = pd.DataFrame(columns=share.columns, index=share.index[0:1])
share_last_month.fillna(0, inplace=True)
for year in range(2008,2019):
    initialCaptial = CAPITAL            # 每年重置初始本金
    for month in range(1,13):
        this_month = str(year)+'-'+str(month)
        
        try:
            temp = round(share[this_month] * initialCaptial/ price,-2)
        except:
            continue
        
        share[this_month] = temp.fillna(method='ffill')
        share_last_day = share[this_month].shift(1)
        share_last_day.iloc[0] = share_last_month.iloc[-1] * price_change.iloc[-1]
        share_last_month = share[this_month]
        # 当日收益 = 昨日share * 今日涨跌幅 ; 每月第一行缺失

        daily_pnl = daily_pnl.append(price_change[this_month] * share_last_day)
        initialCaptial += daily_pnl[this_month].T.sum().sum()
        # print(this_month, initialCaptial)
# 手续费，卖出时一次性收取
fee_rate = 0.0013
fee = (share.diff()[share<share.shift(1)] * priceFill * fee_rate).fillna(0).abs()
daily_pnl -= fee
# 按年清空
cum_pnl = pd.DataFrame(daily_pnl.T.sum(),columns=['pnl'])
cum_pnl['year']=cum_pnl.index.year
cum_pnl = cum_pnl.groupby('year')['pnl'].cumsum()   # 累计收益（金额）
NAV = (daily_pnl.T.sum()/CAPITAL).cumsum()+1        # 净值 按月复利 每年重置 累计值
NAV0 = (cum_pnl / CAPITAL)+1                        # 净值 按月复利 每年重置
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
    
    NAV[str(YEAR)].plot(label='按月复利 每年重置 累计值')
#    NAV0.plot(label='按月复利 每年重置')
    hs300 = hs300/hs300.iloc[0]
    hs300.plot(label='hs300')
#    exec(underLying+' = '+underLying+".reindex(daily_pnl.index)")
#    exec(underLying+' = '+underLying+'/'+underLying+'.iloc[0]')
#    exec(underLying+".plot(label='"+underLying+"')")
    plt.legend(fontsize=11)
    # plt.title('RSI参数={}，日频，无手续费'.format(RSI_arg),fontsize=15)
    plt.title('RSI参数={}，日频，手续费{:.1f}‰'.format(RSI_arg, fee_rate*1000), fontsize=15)
    # plt.title('RSI参数={}，交易日={}，手续费{:.1f}‰'.format(RSI_arg, 交易日+1, fee_rate*1000), fontsize=15)
    plt.grid(axis='both')
    plt.show()
图像绘制()
#%% 输出
posSlow[str(YEAR)].T.sum().plot()
print(YEAR, "RSI:", NAV.iloc[-1]-1, "hs300:", (hs300/hs300.iloc[0]).iloc[-1]-1)
nav0 = NAV[str(YEAR)]
nav0.name = 'NAV'
hs300.name = 'hs300'
pnl=(daily_pnl.T.sum() / CAPITAL)[str(YEAR)]
pnl.name = 'pct_change'
pos = posSlow[str(YEAR)].T.sum()[str(YEAR)]
pos.name='position'
summary = pd.concat([nav0, pnl, hs300.pct_change(), pos], axis=1)
sumall = pd.concat([sumall, summary])
#%% See what happened
def see():
    for index in price.columns:
        posSlow[index]['2009'].plot()
        plt.ylim((-0.001, 0.014))
        plt.twinx()
        price[index]['2009'].plot(c='black')
        plt.title(index)
        plt.show()
        
        daily_pnl[index]['2009'].cumsum().plot(c='gold')
        plt.twinx()
        price[index]['2009'].plot(c='black')
        plt.show()
        
        if input()!='':break
