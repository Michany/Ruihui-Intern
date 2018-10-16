# -*- coding: utf-8 -*-
"""
@author: Michael Wang

- 分别做了HS300和中证500

- 修复了在按周调仓的认定上的bug.  
  根据周频reindex后会有nan值，而随后的fillna会使得na值被向前覆盖。  
  需要先fillna再进行

- 采用前复权价格（更新data_reader)

- 新增近似股数功能

- 新增手续费

updated on 2018/10/16
"""
# pylint: disable=E1101,E1103

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from data_reader import get_muti_close_day, get_index_day
import datetime
import pymssql


underLying = 'hs300'  # zz500
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

data = pd.read_sql(SQL, conn)
pool = list(data['code'])
del data

# 获取数据，计算RSI
START_DATE = '20070201'
END_DATE = datetime.date.today().strftime('%Y-%m-%d')
print('正在获取数据...')
price = get_muti_close_day(pool, START_DATE, END_DATE)
price.fillna(method="ffill", inplace=True)
print("Historical Data Loaded!")

#%% 仓位计算和优化处理
RSI_arg = 30
RSI = price.apply(getattr(talib, 'RSI'), args=(RSI_arg,))
RSI = RSI.replace(0, np.nan)
分母 = abs(RSI.T-50).sum()
RSI_normalized = ((RSI.T-50)/分母).T
RSI_normalized.fillna(0, inplace=True)
pos = RSI_normalized[RSI_normalized > 0]
#pos = pos.multiply((RSI_normalized.T.sum().T + 0.3),axis=0)
#pos[pos<0] = 0
#pos[pos.T.sum()>1] /= pos.T.sum()
<<<<<<< HEAD
pos[pos.T.sum()>0.6] *= 1.1
pos[pos.T.sum()>0.7] *= 1.1
pos[pos.T.sum()>0.8] *= 1.1
pos[pos.T.sum()<0.4] *= 0.8
pos[pos.T.sum()<0.3] *= 0.5
# 将总和超出1的仓位，除以总和，归为1
pos[pos.T.sum()>1] = pos[pos.T.sum()>1].divide(pos.T.sum()[pos.T.sum()>1],axis=0)
pos.fillna(0, inplace = True)

# share记录了实时的仓位信息
# 注意：交易时间为交易日的收盘前
交易日=4
share = pos#[pos.index.dayofweek == 交易日]
share = share.reindex(pos.index)
share.fillna(method='ffill',inplace=True)

#近似
=======
pos[pos.T.sum() > 0.6] *= 1.1
pos[pos.T.sum() > 0.7] *= 1.1
pos[pos.T.sum() > 0.8] *= 1.1
pos[pos.T.sum() < 0.4] *= 0.8
pos[pos.T.sum() < 0.3] *= 0.5
pos[pos.T.sum() > 1] = pos[pos.T.sum() > 1].divide(pos.T.sum()[pos.T.sum() > 1], axis=0)
pos.fillna(0, inplace=True)
# share记录了实时的仓位信息
交易日 = 4
share = pos  # [pos.index.dayofweek == 交易日]
#share[share<1e-4] = 0
share = share.reindex(pos.index)
share.fillna(method='ffill', inplace=True)
# 近似
>>>>>>> a6aaaa1188dbf6e864648ed2c76d1f2809485e9f
initialCaptial = 1e6
that = round(share * initialCaptial / price, -2)
share = that.fillna(method='ffill')
del that
# balance为账户余额
balance = share * price

<<<<<<< HEAD
# 前一天的账户余额 * 当日股价变动 = 当日盈亏
daily_pnl = price_pct_change * balance.shift(1)
# 手续费，卖出时收取
fee_rate = 0.0013
fee = (share.diff()[share<share.shift(1)] * price * fee_rate).fillna(0).abs()
daily_pnl -= fee

#分别绘制复利和单利
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(9,6))

NAV = daily_pnl.T.sum().cumsum() / initialCaptial + 1
NAV.plot(label='单利')
=======
#
price_pct_change = price.pct_change().replace(np.inf, 0)
daily_pnl = price_pct_change * balance.shift(1)
# 分别绘制复利和单利
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(9, 6))
#(daily_pnl.T.sum()+1).cumprod().plot(label='复利')
(daily_pnl.T.sum().cumsum()+1).plot(label='单利')
>>>>>>> a6aaaa1188dbf6e864648ed2c76d1f2809485e9f
hs300 = hs300.reindex(daily_pnl.index)
hs300 /= hs300.iloc[0]
hs300.plot(label='HS300')
plt.legend(fontsize=15)
<<<<<<< HEAD
# plt.title('RSI参数={}，日频，无手续费'.format(RSI_arg),fontsize=15)
plt.title('RSI参数={}，日频，手续费{:.1f}‰'.format(RSI_arg, fee_rate*1000), fontsize=15)
plt.show()

#%% 看回测
df = pd.DataFrame({'Daily_pnl':daily_pnl.T.sum(), 'NAV':(daily_pnl.T.sum().cumsum()+1)},index = daily_pnl.index)
=======
plt.title('RSI参数={}，日频'.format(RSI_arg, 交易日+1), fontsize=15)
plt.show()

#%%看回测
df = pd.DataFrame({'Daily_pnl': daily_pnl.T.sum(), 'NAV': (daily_pnl.T.sum().cumsum()+1)}, 
                    index=daily_pnl.index)
>>>>>>> a6aaaa1188dbf6e864648ed2c76d1f2809485e9f
df.to_excel('RSI横截面_纯多头_收益率明细_{}.xlsx'.format(datetime.date.today().strftime('%y-%m-%d')),
            sheet_name='RSI={},周{},周频'.format(RSI_arg, 交易日+1))
share.to_excel('RSI横截面_纯多头_持仓明细_{}.xlsx'.format(datetime.date.today().strftime('%y-%m-%d')),
               sheet_name='RSI={},周{},周频'.format(RSI_arg, 交易日+1))
# 在draw_pdf.py中画图

print("沪深300年收益")
print(hs300.resample('y').last()/hs300.resample('y').first()-1)
print('2017年收益：{:.2%}'.format(daily_pnl['2017'].T.sum().sum()))
