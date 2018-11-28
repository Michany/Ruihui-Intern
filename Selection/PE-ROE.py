# -*- coding: utf-8 -*-
"""
@author: Michael Wang

PR-ROE回归关系选股
----------------

根据申万宏源研报《PE-ROE股指原理与预期差选股》编写

updated on 2018/11/27
"""

import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pymssql
import talib
import sklearn as skl
from sklearn.linear_model import LinearRegression
from data_reader import get_muti_close_day, get_index_day

# %% 选股
def collect_data():
    global data, industry, pb, t

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
    data.drop_duplicates(subset=['code','REPORT_PERIOD'],inplace=True)
    data['REPORT_PERIOD'] = data['REPORT_PERIOD'].apply(lambda s:s[:4]+'-'+s[4:6]+'-'+s[6:])
    data = data.astype({'REPORT_PERIOD':'datetime64'})
    # 每次获取到财务报表的时间不一，季报的时间统一为报告有效期后的20天，年报的时间统一为报告有效期后的80天
    data['ANN_DT'] = data['REPORT_PERIOD'].apply(lambda x:x+pd.Timedelta(80,'d') if x.is_year_end else x+pd.Timedelta(20,'d'))
    # data.set_index('ANN_DT',inplace=True)

    industry=pd.read_sql('SELECT * FROM AShareIndustriesName',conn)

    print("Data OK\n正在选股...")

    t0=time.time()

    # industry.drop(columns=[''])
    data=data.merge(industry, on='code')
    data.set_index('ANN_DT',inplace=True)

    # 提取PB财务数据
    SQL= '''
    SELECT
        a.S_INFO_WINDCODE as code,
        a.TRADE_DT as date,
        a.S_VAL_PB_NEW as PB
    FROM
        ASHAREEODDERIVATIVEINDICATOR AS a 
    WHERE
        TRADE_DT > '20090101' 
        AND ( TRADE_DT LIKE '____0331' OR TRADE_DT LIKE '____0630' OR TRADE_DT LIKE '____0930' OR TRADE_DT LIKE '____1231' ) 
    ORDER BY
        a.S_INFO_WINDCODE, a.TRADE_DT
    '''
    pb = pd.read_sql(SQL,conn)
    pb['date'] = pb['date'].apply(lambda s:s[:4]+'-'+s[4:6]+'-'+s[6:])
    pb = pb.astype({'date':'datetime64'})
    # price = pd.read_hdf(r"C:\Users\meiconte\Documents\RH\IndexEnhancement\PriceData_1101.h5", 'df')
    # price = price.reindex(pd.date_range('2008-1-2','2018-11-1'), method='ffill')
    # for i in data.index:
    #     data.loc[i, 'price']= price.loc[data['ANN_DT'][i],data['code'][i]]
    # data['PB']=data['price']/data['S_FA_BPS']

    t = pd.concat([data.set_index(['code','REPORT_PERIOD']),pb.set_index(['code','date'])], axis=1)
    t.dropna(inplace=True)
    t=pd.concat([t,t.index.to_frame()],axis=1)
    t.rename(columns={0:'code', 1:'date'}, inplace=True)
    t['lnPB'] = t.PB.apply(np.log)
collect_data()

#%% 分组 回归
def kill_outliers(data, columns):
    
    sigma = data[columns].std()
    mu = data[columns].mean()
    data = data[(data[columns]<mu+3*sigma) & (data[columns]>mu+3*sigma)]

    return data
def group_regression():
    tgroup = t.groupby(['date','c_name'])
    reg = LinearRegression()
    record = pd.DataFrame()
    for index, sector_data in tgroup:
        print("\r正在计算 {} {}".format(*index),end='')
        reg.fit(sector_data['lnPB'].values.reshape(-1, 1), sector_data['ROE'].values.reshape(-1, 1))
        ROE预期差 = sector_data['ROE']-sector_data['lnPB'].apply(reg.predict)
        ROE预期差 = ROE预期差.apply(lambda x:x[0][0]) # 原本x是[[0.002]]的array，现在转化为float
        ROE预期差.name = 'ROE预期差'
        record = pd.concat([record,ROE预期差],axis=0)
        # sn.lmplot('lnPB','ROE',sector_data)
        # plt.show()
        # if input()!='':break
    



# TODO ROE>15,ROE稳定,市值排名前50%或者大于10亿
for firm in g: #利用rolling的方法加快速度，设定window=1000，min_periods=0保证达到累计平均的效果
    print("正在计算 {}".format(firm[0]),end='\r')
    firm[1].sort_index(inplace=True)
    #print(firm[1])
    #break
    temp_rolling = firm[1].S_FA_ROE_TTM.rolling(1000,min_periods=0)
    temp=temp_rolling.mean()
    temp.name = firm[0]
    roe_mean=pd.concat([roe_mean,pd.DataFrame(temp)],sort=False,axis=1)
    temp=temp_rolling.std()<7
    temp.name = firm[0]
    roe_stable=pd.concat([roe_stable,pd.DataFrame(temp)],sort=False,axis=1)

roe_strong = roe_mean>20
choice = roe_strong * roe_stable
choice.sort_index(inplace=True)
choice = choice[self.START_DATE:self.END_DATE] #时间截断
choice.fillna(method='ffill',inplace=True) # 用最新的roe数据，直至新的roe数据出现，所以用ffill
roe_mean=roe_mean.sort_index().fillna(method='ffill')

行业对照表 = roe.drop_duplicates(subset=['code','c_name'])
行业对照表=行业对照表.drop(columns=['S_FA_ROE_TTM','name']).set_index('code')
self.行业对照表=行业对照表

# 根据行业来选
industry_choice=pd.DataFrame(index=choice.index,columns=set(行业对照表.c_name))
industry_choice.fillna(0, inplace=True)

for symbol in choice.columns:
    sector=行业对照表.loc[symbol][1]
    industry_choice.loc[choice[symbol][choice[symbol].isin([1])].index, sector]=\
        industry_choice.loc[choice[symbol][choice[symbol].isin([1])].index, sector].apply(
            lambda x:x+[symbol,] if isinstance(x,list) else [symbol,])
# 每个行业选取最优的
def compare_then_place(exist, day):
    temp_max=-20
    if exist==0:
        return 0
    existSymbols = exist
    for existSymbol in existSymbols:
        if roe_mean.loc[day,existSymbol] > temp_max:
            temp_max = roe_mean.loc[day,existSymbol]
            max_symbol = existSymbol
    return max_symbol

for day in industry_choice.index:
    industry_choice.loc[day]=industry_choice.loc[day].T.apply(compare_then_place,args=(day,))
t1 = time.time()
tpy = t1 - t0
print("\n行业内优选已完成，选股总用时 %5.3f 秒" % tpy)

# 所有曾经选中过的股票都要获取历史数据
s = set()
for sector in industry_choice.columns:
    s=s.union(set(industry_choice[sector]))
s.remove(0)

