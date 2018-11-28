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
    print("P/B Data OK")

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
    data = data[(data[columns]<mu+3*sigma) & (data[columns]>mu-3*sigma)]

    return data

#t = kill_outliers(t, ['ROE','lnPB'])

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
        # sn.lmplot('lnPB','ROE',sector_data)
        # plt.show()
        # if input()!='':break
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
#%% 回测
# share记录了实时的仓位信息
# 交易时间为交易日的收盘前
CAPITAL = 1E6
share = pos[pos>0]
share = share.reindex(price.index)
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
    
    NAV.plot(label='按月复利 每年重置 累计值')
    NAV0.plot(label='按月复利 每年重置')
    exec(underLying+' = '+underLying+".reindex(daily_pnl.index)")
    exec(underLying+' = '+underLying+'/'+underLying+'.iloc[0]')
    exec(underLying+".plot(label='"+underLying+"')")
    plt.legend(fontsize=11)
    # plt.title('RSI参数={}，日频，无手续费'.format(RSI_arg),fontsize=15)
    plt.title('RSI参数={}，日频，手续费{:.1f}‰'.format(RSI_arg, fee_rate*1000), fontsize=15)
    # plt.title('RSI参数={}，交易日={}，手续费{:.1f}‰'.format(RSI_arg, 交易日+1, fee_rate*1000), fontsize=15)
    plt.grid(axis='both')
    plt.show()
图像绘制()






