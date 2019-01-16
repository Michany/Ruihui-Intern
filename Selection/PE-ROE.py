# -*- coding: utf-8 -*-
"""
@author: Michael Wang

PR-ROE回归关系选股
----------------

根据申万宏源研报《PE-ROE股指原理与预期差选股》编写
2018/12/5 电话咨询分析师

TODO
u+-3*σ的剔除函数，不知道是在一个时间横截面上剔除，还是在一个个股的横截面上剔除？
A: 只是剔除了PB>30

在行业分类上，研报中采用了申万二级，我这边采用了我所能找到的Sina一级分类。
A: 细分以后，确实会很少，然后回归线的斜率也是负数，这种情况需要剔除
    
股票池需要进一步细分为中盘，大盘，小盘，但是细分标准需要统一。
A: 细分标准是 前20%大盘股 后30%小盘股

市值>100亿元，在2015年以前
updated on 2018/12/6
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

    conn=pymssql.connect(server='10.0.0.51',port=1433,user='sa',password='abc123',database='WIND') 
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
    # updated: 剔除PB>100的情况
    pb = pb[pb.PB<100] 
    mv = pb.pivot_table(values='MV', columns='code', index='date')
    print("P/B, Market Value Data OK")

    t = pd.concat([pb.set_index(['code','date']), data.set_index(['code','REPORT_PERIOD'])], axis=1)
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

#t = kill_outliers(t, ['ROE','lnPB'])

def group_regression(mute = True):
    tgroup = t.groupby(['date','c_name'])
    reg = LinearRegression()
    record = pd.DataFrame()
    for index, sector_data in tgroup:
        print("\r正在回归 {} {}".format(*index),end='')

        # kill_outliers(sector_data, ['ROE','lnPB'])
        reg.fit(sector_data['lnPB'].values.reshape(-1, 1), sector_data['ROE'].values.reshape(-1, 1))
        # 如果斜率小于0，则把这种情况丢弃
        if reg.coef_ < 0:
            continue
        ROE预期差 = sector_data['ROE']-sector_data['lnPB'].apply(reg.predict)
        ROE预期差 = ROE预期差.apply(lambda x:x[0][0]) # 原本x是[[0.002]]的array，现在转化为float
        ROE预期差.name = 'ROE预期差'            
        record = pd.concat([record,ROE预期差],axis=0)
        if not mute:
            sn.lmplot('lnPB','ROE',sector_data)
            plt.show()
            if input()!='':break
    return record

t0 = time.time()
ROE_diff = group_regression()
ROE_diff['index']=ROE_diff.index
ROE_diff['code'] = ROE_diff['index'].apply(lambda x:x[0])
ROE_diff['date'] = ROE_diff['index'].apply(lambda x:x[1])
ROE_diff = ROE_diff.drop(columns=['index'])
selection = ROE_diff.pivot_table(values=0,index='date',columns='code')
selection = selection.fillna(0)

tpy = time.time() - t0
print("\n行业内优选已完成，选股总用时 %5.3f 秒" % tpy)

#%% 获取价格数据
price = get_muti_close_day(selection.columns, '2009-03-31', '2018-11-30', freq = 'M', adjust=-1) # 回测时还是使用前复权价格
priceFill = price.fillna(method='ffill') 
price_change = priceFill.diff()
hs300 = get_index_day('000300.SH','2010-4-30','2018-11-30','M').sclose
szzz = get_index_day('000001.SH','2009-4-30','2018-11-30','M').sclose
zz500 = get_index_day('000905.SH','2009-4-30','2018-11-30','M').sclose

IF = pd.read_excel("IF.xlsx").set_index('date')
#IC = pd.read_sql("SELECT date, settle1 FROM DCindexfutures WHERE name like 'IC'", conn, index_col='date')
IF['CLOSE']=IF['IF']
#IC['CLOSE']=IC['settle1']
#IC = pd.read_hdf("monthlyData-over10B.h5",'IC')
#IF = pd.read_hdf("monthlyData-over10B.h5",'IF')
# 剔除没有价格数据的部分股票
l=list(mv.columns)
for i in price.columns:
    l.remove(i)
mv = mv.drop(columns=l)
#%% 股票选择
isBig = (mv.T > mv.T.quantile(0.8)).T
isSmall = (mv.T < mv.T.quantile(0.3)).T
def reindex_fill_gap(inputDataFrame, defaultIndex = price.index):
    return inputDataFrame.reindex(defaultIndex, method='ffill')
selection = reindex_fill_gap(selection)
isBig = reindex_fill_gap(isBig)
isSmall = reindex_fill_gap(isSmall)
#%% 回测
pos = (mv.T/mv.T.sum()).T #按照市值加权作为仓位
pos = pos.reindex(price.index, method='ffill')

pos = pos[selection>0]# 选取selection中结果大于零的

大盘股=1
if 大盘股:
    pos = pos[isBig==True]# 选取大盘股
    pos = (pos.T/(pos.T.sum())).T# 将仓位调整至100%
else:
    pos = pos[isSmall==True]# 选取小盘股
    pos_col = 1/((pos>0).T.sum())
    for col in pos.columns:  # 小盘股等值加权
        pos[col]=pos_col     # 讲等权值赋给每一行
    pos = pos[selection>0]   # 重新剔除
    pos = pos[isSmall==True] # 重新剔除

# 计算当日盈亏百分比
daily_pnl = pos * priceFill.pct_change()

daily_pnl = daily_pnl['2010-04-30':]#['2015-04-16':]
NAV = (daily_pnl.T.sum()+1).cumprod() #计算净值
NAV0 = 1+(daily_pnl.T.sum()).cumsum() #计算净值

IC = IC.resample('M').last()
IF = IF.resample('M').last()

#画图
plt.figure(figsize=(8,6))
NAV.plot(label='Selection')
if 大盘股:
    (IF.CLOSE.pct_change()+1).cumprod().plot(label='000300.SH')
    (NAV/(IF.CLOSE.pct_change()+1).cumprod()).plot(label='Exess Return')
else:
    (IC.CLOSE.pct_change()+1).cumprod().plot(label='000905.SH')
    (NAV/(IC.CLOSE.pct_change()+1).cumprod()).plot(label='Exess Return')
plt.legend(fontsize=14)


#%%
def check(y):#看一下具体每一年的表现
    year = str(y)
#    (NAV[year]/NAV[year].iloc[0]).plot()
#    (hs300[year]/hs300[year].iloc[0]).plot(c='black')
#    plt.show()
    print(y, (NAV[year]/NAV[year].iloc[0]).iloc[-1]-1,(hs300[year]/hs300[year].iloc[0]).iloc[-1]-1)

for i in range(2010,2019):
    check(i)
temp = (daily_pnl.T.sum()-hs300.pct_change()).fillna(0)
plt.hist(temp)
#%%
def excel_output():
    excess = (daily_pnl.T.sum()-IF.CLOSE.pct_change()).cumsum()+1
    excess.name = 'NAV'
    excess.to_excel('excess return.xlsx')
#excel_output()
