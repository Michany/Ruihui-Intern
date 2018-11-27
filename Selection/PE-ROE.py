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
import talib
from data_reader import get_muti_close_day, get_index_day, get_hk_index_day
import pymssql

# %% 选股
def choosing(self):
    conn=pymssql.connect(server='192.168.0.28',port=1433,user='sa',password='abc123',database='WIND') 
    SQL='''
    SELECT b.code, a.S_FA_ROE_TTM, a.s_fa_totalequity_mrq, a.ANN_DT, a.REPORT_PERIOD
    FROM AShareTTMHis as a, HS300COMPWEIGHT as b
    where b.[Date] BETWEEN '2018-07-01' and '2018-07-03' and a.S_INFO_WINDCODE = b.code 
    ORDER BY b.code, a.REPORT_PERIOD'''#, a.s_fa_totalequity_mrq    and a.REPORT_PERIOD LIKE '____1231'
    data = pd.read_sql(SQL,conn)
    data.dropna(subset=['S_FA_ROE_TTM'], inplace=True)
    data.drop_duplicates(subset=['code','REPORT_PERIOD'],inplace=True)
    # data.dropna(subset=['ANN_DT'], inplace=True)
    data.REPORT_PERIOD = data.REPORT_PERIOD.apply(lambda s:s[:4]+'-'+s[4:6]+'-'+s[6:])
    data = data.astype({'REPORT_PERIOD':'datetime64'})
    data['ANN_DT'] = data['REPORT_PERIOD'].apply(lambda x:x+pd.Timedelta(80,'d') if x.is_year_end else x+pd.Timedelta(20,'d'))

    # data.set_index('ANN_DT',inplace=True)
    
    industry=pd.read_sql('SELECT * FROM AShareIndustriesName',conn)

    print("Data OK\n正在选股...")

    t0=time.time()
    # industry.drop(columns=[''])
    roe=data.merge(industry, on='code')
    roe.set_index('ANN_DT',inplace=True)
    # TODO ROE>15,ROE稳定,市值排名前50%或者大于10亿
    g = roe.groupby('code')
    roe_strong=pd.DataFrame()
    roe_stable=pd.DataFrame()
    roe_mean=pd.DataFrame()
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
    self.industry_choice=industry_choice
    t1 = time.time()
    tpy = t1 - t0
    print("\n行业内优选已完成，选股总用时 %5.3f 秒" % tpy)

    # 所有曾经选中过的股票都要获取历史数据
    s = set()
    for sector in industry_choice.columns:
        s=s.union(set(industry_choice[sector]))
    s.remove(0)
    self.chosed_pool = s
