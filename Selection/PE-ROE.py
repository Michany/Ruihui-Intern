# -*- coding: utf-8 -*-
"""
@author: Michael Wang

PR-ROE回归关系选股
----------------

根据申万宏源研报《PE-ROE股指原理与预期差选股》编写

TODO
u+-3*σ的剔除函数，不知道是在一个时间横截面上剔除，还是在一个个股的横截面上剔除？

在行业分类上，研报中采用了申万二级，我这边采用了我所能找到的Sina一级分类。

股票池需要进一步细分为中盘，大盘，小盘，但是细分标准需要统一。

市值
updated on 2018/12/3
"""

import datetime
import time
from data_reader import get_index_day, get_muti_close_day

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn as skl
import talib
from sklearn.linear_model import LinearRegression

import pymssql


# %% 选股
def collect_data():
    global data, industry, pb, mv, t

    conn=pymssql.connect(server='192.168.0.28',port=1433,user='sa',password='abc123',database='WIND') 
    SQL='''
    SELECT
        a.S_INFO_WINDCODE as code,
        a.S_FA_ROE as ROE,
        a.ANN_DT,
        a.REPORT_PERIOD 
    FROM
        AShareFinancialIndicator AS a 
    WHERE
        a.REPORT_PERIOD > '20090101' 
    ORDER BY
        a.S_INFO_WINDCODE,a.REPORT_PERIOD'''#, a.s_fa_totalequity_mrq    and a.REPORT_PERIOD LIKE '____1231'
    data = pd.read_sql(SQL,conn)

    data.dropna(subset=['ROE'], inplace=True)

    data['REPORT_PERIOD'] = data['REPORT_PERIOD'].apply(lambda s:s[:4]+'-'+s[4:6]+'-'+s[6:])
    data = data.astype({'REPORT_PERIOD':'datetime64'})
    # 每次获取到财务报表的时间不一，季报的时间统一为报告有效期后的20天，年报的时间统一为报告有效期后的80天
    data['ANN_DT'] = data['REPORT_PERIOD'].apply(lambda x:x+pd.Timedelta(80,'d') if x.is_year_end else x+pd.Timedelta(20,'d'))
    # data.set_index('ANN_DT',inplace=True)

    industry=pd.read_sql('SELECT * FROM AShareIndustriesName',conn)

    print("Data OK")

    data=data.merge(industry, on='code')
    data.set_index('ANN_DT',inplace=True)
    data = data.drop_duplicates(subset=['code','REPORT_PERIOD'])
    
    # 提取PB财务数据
    SQL= '''
    SELECT
        a.S_INFO_WINDCODE as code,
        a.TRADE_DT as date,
        a.S_VAL_PB_NEW as PB,
        a.S_DQ_MV as MV
    FROM
        ASHAREEODDERIVATIVEINDICATOR AS a 
    WHERE
        TRADE_DT > '20090101' 
        AND S_INFO_WINDCODE IN (SELECT S_INFO_WINDCODE FROM ASHAREEODDERIVATIVEINDICATOR WHERE S_DQ_MV>1000000)
        AND ( TRADE_DT LIKE '____0331' OR TRADE_DT LIKE '____0630' OR TRADE_DT LIKE '____0930' OR TRADE_DT LIKE '____1231' ) 
    ORDER BY
        a.S_INFO_WINDCODE, a.TRADE_DT
    '''
    pb = pd.read_sql(SQL,conn)
    pb['date'] = pb['date'].apply(lambda s:s[:4]+'-'+s[4:6]+'-'+s[6:])
    pb = pb.astype({'date':'datetime64'})
    mv = pb.pivot_table(values='MV', columns='code', index='date')
    print("P/B, Market Value Data OK")

    t = pd.concat([data.set_index(['code','REPORT_PERIOD']),pb.set_index(['code','date'])], axis=1)
    t.dropna(inplace=True)
    t=pd.concat([t,t.index.to_frame()],axis=1)
    t.rename(columns={0:'code', 1:'date'}, inplace=True)
    t['lnPB'] = t.PB.apply(np.log)
collect_data()

#%% 分组 回归
def kill_outliers(data, columns):
    '''
    去除 3 sigma 极端值
    但是有一个问题是，还不能做到同时去除，只能一个个变量一次次去除，这样会导致去除的偏多。
    '''
    ranging = 2
    sigma = dict()
    mu = dict()
    for col in columns:
        sigma[col] = data[col].std()
        mu[col] = data[col].mean()
    for col in columns:
       data = data[(data[col]<mu[col]+ranging*sigma[col]) & (data[col]>mu[col]-ranging*sigma[col])]

    return data

t = kill_outliers(t, ['ROE','lnPB'])

def group_regression():
    tgroup = t.groupby(['date','c_name'])
    reg = LinearRegression()
    record = pd.DataFrame()
    for index, sector_data in tgroup:
        print("\r正在回归 {} {}".format(*index),end='')

        kill_outliers(sector_data, ['ROE','lnPB'])
        reg.fit(sector_data['lnPB'].values.reshape(-1, 1), sector_data['ROE'].values.reshape(-1, 1))

        ROE预期差 = sector_data['ROE']-sector_data['lnPB'].apply(reg.predict)
        ROE预期差 = ROE预期差.apply(lambda x:x[0][0]) # 原本x是[[0.002]]的array，现在转化为float
        ROE预期差.name = 'ROE预期差'
        record = pd.concat([record,ROE预期差],axis=0)
#        sn.lmplot('lnPB','ROE',sector_data)
#        plt.show()
#        if input()!='':break
    return record

t0 = time.time()
ROE_diff = group_regression()
ROE_diff['index']=ROE_diff.index
ROE_diff['code'] = ROE_diff['index'].apply(lambda x:x[0])
ROE_diff['date'] = ROE_diff['index'].apply(lambda x:x[1])
ROE_diff = ROE_diff.drop(columns=['index'])
selection = ROE_diff.pivot_table(values=0,index='date',columns='code')

# pos = pos[pos>0]

tpy = time.time() - t0
print("\n行业内优选已完成，选股总用时 %5.3f 秒" % tpy)

#%% 获取价格数据
price = get_muti_close_day(selection.columns, '2009-03-31', '2018-11-30', freq = 'M', adjust=-1) # 回测时还是使用前复权价格
priceFill = price.fillna(method='ffill') 
price_change = priceFill.diff()
hs300 = get_index_day('000001.SH','2009-4-30','2018-11-30','M').sclose
#%% 回测
CAPITAL = 1E6
pos = (mv.T/mv.T.sum()).T #按照市值加权作为仓位

pos = pos.reindex(price.index, method='ffill')
daily_pnl = pos * priceFill.pct_change()
NAV = (daily_pnl.T.sum()+1).cumprod()

plt.figure(figsize=(8,6))
NAV.plot(label='Selection')
(hs300.pct_change()+1).cumprod().plot(label='000001.SH')
(NAV/(hs300.pct_change()+1).cumprod()).plot(label='Exess Return')
plt.legend(fontsize=14)

#%%
def 图像绘制():
    global hs300
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(9,6))
    
    NAV.plot(label='按月复利 每年重置 累计值')

    plt.legend(fontsize=11)
    # plt.title('RSI参数={}，日频，无手续费'.format(RSI_arg),fontsize=15)
    plt.title('RSI参数={}，日频，手续费{:.1f}‰'.format(RSI_arg, fee_rate*1000), fontsize=15)
    # plt.title('RSI参数={}，交易日={}，手续费{:.1f}‰'.format(RSI_arg, 交易日+1, fee_rate*1000), fontsize=15)
    plt.grid(axis='both')
    plt.show()
图像绘制()
