# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:23:45 2018

@author: Michael Wang

- 分别做了HS300和中证500

updated on 2018/10/9
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from data_reader import get_muti_close_day, get_index_day

#pool = ['600309.SH', '600585.SH', '000538.SZ', '000651.SZ', '600104.SH', '600519.SH', '601888.SH']
# pool=['600000.SH','600016.SH','600019.SH','600028.SH','600029.SH','600030.SH','600036.SH','600048.SH','600050.SH','600104.SH','600111.SH','600309.SH','600340.SH','600518.SH','600519.SH','600547.SH','600606.SH','600837.SH','600887.SH','600919.SH','600958.SH','600999.SH','601006.SH','601088.SH','601166.SH','601169.SH','601186.SH','601211.SH','601229.SH','601288.SH','601318.SH','601328.SH','601336.SH','601390.SH','601398.SH','601601.SH','601628.SH','601668.SH','601669.SH','601688.SH','601766.SH','601800.SH','601818.SH','601857.SH','601878.SH','601881.SH','601985.SH','601988.SH','601989.SH','603993.SH']
try:
    import pymssql
except:
    raise Exception("请安装所需包: 打开命令提示符cmd，输入 pip install pymssql 安装")
conn=pymssql.connect(server='192.168.0.28',port=1433,user='sa',password='abc123',
                     database='RawData') 
SQL='''
SELECT b.code FROM HS300COMPWEIGHT as b
where b.[Date] BETWEEN '2018-07-01' and '2018-07-03' ORDER BY b.code'''#, a.s_fa_totalequity_mrq    and a.REPORT_PERIOD LIKE '____1231'
SQL='''
SELECT b.code FROM [中证500成分权重] as b
where b.[Date] BETWEEN '2018-06-21' and '2018-07-03'
'''
data = pd.read_sql(SQL,conn)
pool = list(data['code'])

hs300 = get_index_day('000300.SH','2007-02-01','2018-08-30','1D')
hs300 = hs300.sclose


# 获取数据，计算RSI
START_DATE = '20070201'
END_DATE = '20180830'
price = get_muti_close_day(pool, START_DATE, END_DATE)
price.fillna(method="ffill", inplace=True)
print("Historical Data Loaded!")

#%%
RSI_arg = 25
RSI = price.apply(getattr(talib, 'RSI'), args=(RSI_arg,))
RSI=RSI.replace(0,np.nan)
分母=abs(RSI.T-50).sum()
RSI_normalized = ((RSI.T-50)/分母).T
RSI_normalized.fillna(0,inplace=True)
pos = RSI_normalized[RSI_normalized>0]

# share记录了实时的仓位信息
交易日=4
share = pos[pos.index.dayofweek == 交易日]
share = share.reindex(pos.index)
share = share.fillna(method='ffill')

price_pct_change = price.pct_change().replace(np.inf,0)
daily_pnl = price_pct_change.shift(-1) * share
#分别绘制复利和单利
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(9,6))
#(daily_pnl.T.sum()+1).cumprod().plot(label='复利')
(daily_pnl.T.sum().cumsum()+1).plot(label='单利')
hs300 = hs300.reindex(daily_pnl.index)
hs300/=hs300.iloc[0]
hs300.plot(label='HS300')
plt.legend(fontsize=15)
plt.title('RSI参数={}，交易日期=周{},日频'.format(RSI_arg,交易日+1),fontsize=15)
plt.show()

#看回测
df = pd.DataFrame({'Daily_pnl':daily_pnl.T.sum(), 'NAV':(daily_pnl.T.sum().cumsum()+1)},index = daily_pnl.index)
df.to_excel('RSI横截面_纯多头.xlsx')
# 在draw_pdf.py中画图
hs300.resample('y').last()/hs300.resample('y').first()-1



