import sys 
sys.path.append(r"C:\Users\meiconte\Documents\RH\Summary")

import pandas as pd
import numpy as np
import summary_AIP_index as s
import pymssql

INDEX_POOL = ['000016.SH','000905.SH','000009.SH','000991.SH','000935.SH','000036.SH']
INDEX_PE_POOL = ['SZ50', 'ZZ500', '000009_SH', '000991_SH', '000935_SH']
FILEPATH = r"C:\Users\meiconte\Documents\RH\excel\TSoutput\\"

worths, index_close = {}, {}
for index in INDEX_POOL:
    print("----- Now processing {} ...".format(index))
    s.read_files(FILEPATH + index + '.xlsx')
    s.start()
    s.AIP_net()
    s.generate_profit_curve()
    print("----- Processed successfully! -----", end='\n\n')
    worths[index]=s.ans
    index_close[index]=s.index_close.sclose
start_date = worths[index].index[0]

'''#%% 合成SQL取数据命令
def get_findata(SQL):
    conn=pymssql.connect(server='192.168.0.28', port=1433, user='sa', password='abc123', database='rawdata',charset = 'GBK')
    data = pd.read_sql(SQL,conn)
    return data

sql_columns = ""
for c in INDEX_PE_POOL: 
    sql_columns += "\"PE_TTM_"+ c + "\","
sql_columns += "\""
sql = "select \"Date\",{} from dbo.Zindexes_PE".format(sql_columns[:-2])
print(sql)

# 取得PE数据
PE = get_findata(sql)

PE.set_index('Date', inplace = True)
pe = PE[PE.index>start_date]
pe = pe.resample('1W').last()
indexes = list(pe.columns)

#%%  计算相对PE
rpe={}
for index in INDEX_PE_POOL: 
    temp=pe["PE_TTM_"+index]/pe["PE_TTM_"+index][0]
    rpe[index] = temp
rpe=pd.DataFrame(rpe)
for index in INDEX_PE_POOL: 
    rpe.loc[5:,index]=pe["PE_TTM_"+index]/pe["PE_TTM_"+index].shift(5)'''
    
#%% 获取定投收盘数据
close = dict()
for index in INDEX_POOL: #最后一个暂时不做
    # 关键净值公式
    '''
    >>> ans.columns
    >>> ['floating_profit', 'comfirmed_profit', 'accumulated_profit', 'accumulated_investment', '当日盈亏'],
    '''
    ans = worths[index]
    this = (ans.当日盈亏.shift(-1)/ans.accumulated_investment)
    #模拟净值 = (this.apply(lambda x: 0 if (x>0.5)or(x<-0.5)or(x==np.nan) else x)+1).cumprod()
    模拟净值 = (this.apply(lambda x: 0 if (x>0.05)or(x<-0.05)or(x==np.nan) else x)).cumsum()+1
    close[index] = 模拟净值.fillna(method='ffill')
close = pd.DataFrame(close, index=ans.index[1:-1])

#%% 计算相对强弱指数RSI
import talib
Duration = 30
rsi={}
for index in INDEX_POOL:
    rsi[index] = talib.RSI(index_close[index],30)
rsi = pd.DataFrame(rsi)

#%% 分配仓位
index_amount = 3
rank=rsi.rank(axis=1, ascending=False)

# 赋权重(仓位)
pos=rsi.copy()
longer_pos = rsi.copy()
for index in INDEX_POOL:
    pos[index] = rank[index].apply(lambda x: 1/index_amount if x<=index_amount else 0) 

#仓位时间调整：原本仓位信息在每周日，调整至每周四
pos.index -= pd.Timedelta(3,'d')
pos = pos.reindex(index=close.index).fillna(method='ffill')


#%% #计算收益
profit, cum_profit= close.copy(), dict()
for index in INDEX_POOL:
    profit[index] = (close[index].shift(-1)/close[index] - 1) * pos[index]

cum_profit['single'] = 1 + (profit.T.sum()).cumsum() 
cum_profit['compound'] = (profit.T.sum()+1).cumprod() 

#%% 绘制最终收益率曲线
# 简易版
#cum_profit['single'].plot(c='gold',alpha=0.9,figsize=(12,10),linewidth=3)
#cum_profit['compound'].plot(c='orange',alpha=0.9,linewidth=3)
cum_profit = pd.DataFrame(cum_profit, index=profit.index)
# 最终版
def generate_profit_curve(ans: pd.DataFrame, column='single'):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ans['NAV'] = ans[column]; ans['Daily_pnl'] = ans[column].diff()
 
    ax = fig.add_subplot(211)
    ax.plot(ans.index, ans['NAV'], linewidth=2, label='净值')
    ax.fill_between(ans.index, ans['NAV'], y2=1,
                    where=(ans['NAV'] < ans['NAV'].shift(1)) |
                    ((ans['NAV'] > ans['NAV'].shift(-1)) &
                     (ans['NAV'] >= ans['NAV'].shift(1))),
                    facecolor='grey',
                    alpha=0.3)
    # 最大回撤标注
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax.legend(fontsize=15)
    plt.grid()

    bx = fig.add_subplot(212)
    width = 1
    if len(ans)>30: width = 1.5
    bx.bar(ans.index, ans['Daily_pnl'].where(ans['Daily_pnl'] > 0),
           width, label='当日盈亏+', color='red', alpha=0.8)
    bx.bar(ans.index, ans['Daily_pnl'].where(ans['Daily_pnl'] < 0),
           width, label='当日盈亏-', color='green', alpha=0.8)
    bx.legend(fontsize=15)
    plt.grid()
generate_profit_curve(cum_profit)