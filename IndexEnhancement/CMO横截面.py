# -*- coding: utf-8 -*-
"""
@author: Michael Wang

`Based on Version CMO`

updated on 2018/11/20
"""
#%%
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
TODAY = datetime.date.today().strftime('%Y-%m-%d')

# 获取标的数据
underLying = 'hs300'#zz500
if underLying == 'hs300':
    conn = pymssql.connect(server='192.168.0.28', port=1433, user='sa', password='abc123', database='WIND')
    SQL = '''SELECT b.code FROM HS300COMPWEIGHT as b
    where b.[Date] BETWEEN '2018-07-01' and '2018-07-03' ORDER BY b.code'''
    hs300 = get_index_day('000300.SH', '2007-02-01', TODAY, '1D')
    hs300 = hs300.sclose
elif underLying == 'zz500':
    conn = pymssql.connect(server='192.168.0.28', port=1433, user='sa', password='abc123', database='RawData')
    SQL = '''SELECT b.code FROM [中证500成分权重] as b
    where b.[Date] BETWEEN '2018-06-21' and '2018-07-03' '''
    zz500 = get_index_day('000905.SH', '2007-02-01', TODAY, '1D')
    zz500 = zz500.sclose
elif underLying == 'hsi':
    hsi = get_hk_index_day('HSI.HI', '2007-02-01', TODAY, '1D')
    hsi = hsi.sclose
elif underLying == 'zz800':
    conn1 = pymssql.connect(server='192.168.0.28', port=1433, user='sa', password='abc123', database='WIND')
    SQL1 = '''SELECT b.code FROM HS300COMPWEIGHT as b where b.[Date] BETWEEN '2018-07-01' and '2018-07-03' ORDER BY b.code'''
    conn2 = pymssql.connect(server='192.168.0.28', port=1433, user='sa', password='abc123', database='RawData')
    SQL2 = '''SELECT b.code FROM [中证500成分权重] as b where b.[Date] BETWEEN '2018-06-21' and '2018-07-03' '''
    zz800 =  get_index_day('000906.SH', '2007-02-01', TODAY, '1D')
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

#%% 
def 获取数据():
    START_DATE = '2007-02-01'
    END_DATE = TODAY
    print('正在获取数据...自 {} 至 {}'.format(START_DATE, END_DATE))
    price = get_muti_close_day(pool, START_DATE, END_DATE, HK=(underLying=='hsi'))
    priceFill = price.fillna(method="ffill")
    print("Historical Data Loaded!")
    return price, priceFill
price, priceFill = 获取数据()
#%%
def 仓位计算和优化(arg=30, fast = False):
    global CMO_arg, CMO

    CMO_arg = arg
    CMO = priceFill.apply(talib.CMO, args=(CMO_arg,))

    CMO[price.isna()] = 0

    分母=abs(CMO.T).sum()
    CMO_normalized = ((CMO.T)/分母).T
    CMO_normalized.fillna(0,inplace=True)
    pos = CMO_normalized[CMO_normalized>0]

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

posSlow = 仓位计算和优化(14)

def check(y):#看一下具体每一年的表现
    year = str(y)
    posSlow[year].T.sum().plot(c='r')
    plt.twinx()
    hs300[year].plot()
    plt.show()
    
    NAV0[year].plot(c='r')
    (hs300[year]/hs300[year].iloc[0]).plot()
    plt.show()
    
    CMO.T.sum()[year].plot()


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
        # 当日收益 = 昨日share * 今日涨跌幅
        daily_pnl = daily_pnl.append(price_pct_change[this_month] * (share[this_month]*price[this_month]).shift(1))
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
    plt.title('RSI参数={}，日频，手续费{:.1f}‰'.format(CMO_arg, fee_rate*1000), fontsize=15)
    # plt.title('RSI参数={}，交易日={}，手续费{:.1f}‰'.format(RSI_arg, 交易日+1, fee_rate*1000), fontsize=15)
    plt.grid(axis='both')
    plt.show()
图像绘制()