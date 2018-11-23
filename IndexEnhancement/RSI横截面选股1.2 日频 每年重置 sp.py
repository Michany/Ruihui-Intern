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
updated on 2018/11/23
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
CAPITAL = 1e6
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

# 
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
    global RSI_arg, RSI

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
#%% 
def excel输出():
    df = pd.DataFrame({'Daily_pnl':daily_pnl.T.sum(),
                       '累计PNL':cum_pnl,
                       '账户价值':cum_pnl+CAPITAL,
                       'NAV':NAV0, 'NAV累计':NAV},
                       index = daily_pnl.index)
    df.index.name = 'date'
    df.to_excel('RSI横截面_{}纯多头_收益率明细_{}_日.xlsx'.format(underLying, TODAY),
                sheet_name = 'RSI={},日频'.format(RSI_arg))
    
    df = daily_pnl.join(share, lsuffix='_pnl',rsuffix='_share')
    df = df.join(price,rsuffix='_price')
    df = df.join(RSI_Fast, rsuffix='_RSI_Fast')
    df = df.join(RSI_Slow, rsuffix='_RSI_Slow')
    df.sort_index(axis=1,inplace=True)
    df.columns = pd.MultiIndex.from_product([daily_pnl.columns,['price','RSI_Fast','RSI_Slow','daily_pnl','share']])
    df.to_excel('RSI横截面_{}纯多头_持仓明细_{}_日.xlsx'.format(underLying, TODAY),
                sheet_name = 'RSI={},日频'.format(RSI_arg))
excel输出()

#%% 获取实时数据
from WindPy import *
w.start()
data_today = pd.DataFrame()
# 需要加快数据提取速度，改为每次提取10个
batch = 10
batchNo = len(pool)//10
batch_data = pd.DataFrame()
for i in range(batchNo):
    symbol = ""
    for j in range(batch):
        symbol += pool[i*10+j] + ','
    symbol = symbol[:-1] # 去掉最后的逗号
    print("Fetching Data for", symbol)
    rawdata = w.wsi(symbol, "close", "%s 14:55:00" % TODAY, "%s 14:56:00" % TODAY, "")
    rawdata = pd.DataFrame({rawdata.Data[0][0]:rawdata.Data[2]},index=rawdata.Data[1])
    # rawdata 样例：
    #            2018-11-23 14:54:00
    # 000001.SZ                10.32
    # 000002.SZ                24.87
    # 000060.SZ                 4.14
    # 000063.SZ                19.81
    # 000069.SZ                 6.11
    batch_data = pd.concat([batch_data, rawdata.T], axis=1)
data_today = pd.concat([data_today, batch_data], axis=1)
data_close = data_today[data_today.index.minute==55]
data_close = data_close.astype(float)
# 将新老数据拼接起来
price = pd.concat([price, data_close], sort=False)
priceFill = price.fillna(method='ffill')
print("New Data Loaded!", TODAY)

# 计算新仓位
posSlow, RSI_Slow = 仓位计算和优化(40)
posFast, RSI_Fast = 仓位计算和优化(10, fast=True)
posSlow[(posSlow.T.sum()<0.50) & (posSlow.T.sum()>0.05)] = posFast
posSlow[(posSlow.T.sum()>0.95) & (posFast.T.sum()<0.32)] = posFast

# 计算新持股数
share = round(posSlow * initialCaptial/ price, -2)

# 交易信号 
signal = share.diff().iloc[-1]
signal.dropna(inplace=True)
signal = signal[signal!=0]

# 将交易信号文件写入扫单文件csv
def generate_csv_file(扫单软件='cats'):
    if 扫单软件=='cats':
        csv = pd.DataFrame(columns=['下单指令','账户类型','账户','标的代码','委托数量','委托方向','委托价格','委托类型','委托参数（可选）'])
        for i in range(len(signal)):
            symbol = signal.index[i]
            amount = int(signal[symbol])
            委托方向 = int(amount<0)+1
            委托价格 = price[symbol].iloc[-1]
            exchange_type = (symbol[-2:]=='SZ') +1
            委托类型 = 'R' if exchange_type==1 else 'U'
            csv.loc[i] = ['O', 'S0', 'RSI', symbol, 
                          abs(amount), 委托方向, 委托价格, 委托类型, '']
    elif 扫单软件=='other':
        csv = pd.DataFrame(columns=['local_entrust_no','fund_account','exchange_type','stock_code','entrust_bs','entrust_prop','entrust_price','entrust_amount','batch_no'])
        for i in range(len(signal)):
            symbol = signal.index[i]
            amount = int(signal[symbol])
            exchange_type = (symbol[-2:]=='SZ') +1
            entrust_bs = int(amount<0)+1
            entrust_prop = 'R' if exchange_type==1 else 'U'
            csv.loc[i] = [i+1, 7047709, exchange_type, symbol[:-3], 
                        entrust_bs, entrust_prop, price[symbol].iloc[-1],
                        abs(amount), '']
    return csv
csv = generate_csv_file()

# 将扫单文件存入交易主机
import datetime
csv.to_csv(r'\\192.168.0.29\Stock\orders\RSI\order_{}.{}00000.csv'.format('RSItest', datetime.datetime.now().strftime('%Y%m%d%H%M'))
            , header=False, index=False)

#%% 模拟盘PnL跟踪
import openpyxl
tracing_file = openpyxl.load_workbook("模拟盘PnL跟踪.xlsx")
tracing_sheet = tracing_file.active
today_share_record = [share.iloc[-1].name] # 日期时间
today_share_record += list(share.iloc[-1].values) # 持仓数
tracing_sheet.append(list(today_share_record))
tracing_file.save("模拟盘PnL跟踪.xlsx")
