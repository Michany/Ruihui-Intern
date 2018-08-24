import sys 
sys.path.append(r"C:\Users\meiconte\Documents\RH\Summary")

import pandas as pd
import numpy as np
import summary_AIP_stock as s

STOCK_POOL = ['600048.SH','600309.SH','600585.SH','000538.SZ','000651.SZ','600104.SH','600519.SH','601888.SH']

FILEPATH = r"C:\Users\meiconte\Documents\RH\excel\TSoutput_stock\\"

worths, stock_close = {}, {}
for index in STOCK_POOL:
    print("----- Now processing {} ...".format(index))
    s.read_files(FILEPATH + index + '.xlsx')
    s.start()
    s.AIP_net()
    s.generate_profit_curve()
    print("----- Processed successfully! -----", end='\n\n')
    worths[index]=s.ans
    stock_close[index]=s.stock_close.sclose
start_date = worths[index].index[0]

#%% 计算相对强弱指数RSI
import talib
Duration = 30
rsi={}
for index in STOCK_POOL:
    rsi[index] = talib.RSI(stock_close[index].shift(-1),30)
rsi = pd.DataFrame(rsi)


#%% 获取定投收盘数据
close = dict()
# 由于每个标的开始日期不同，需要记录最早的开始日期
earlist_date = worths[STOCK_POOL[0]].index[0]
longest_stock_index = STOCK_POOL[0]
for index in STOCK_POOL: 
    # 关键：净值公式
    ans = worths[index]
    this = (ans.当日盈亏.shift(-1)/ans.accumulated_investment)
    #模拟净值 = (this.apply(lambda x: 0 if (x>0.05)or(x<-0.05)or(x==np.nan) else x)+1).cumprod()
    模拟净值 = (this.apply(lambda x: 0 if (x>0.05)or(x<-0.05)or(x==np.nan) else x)).cumsum()+1
    close[index] = 模拟净值.fillna(method='ffill')
    if ans.index[0] < earlist_date:
        longest_stock_index = index
close = pd.DataFrame(close, index=worths[longest_stock_index].index)
close.plot(figsize=(10,6))


#%% 分配仓位
# 总共买3个标的
amount = 3
# 关键：RSI越高越买，排序时需要用 ascending=False
rank=rsi.rank(axis=1,ascending=False)

# 赋权重(仓位)
pos=rsi.copy()
longer_pos = rsi.copy()
for index in STOCK_POOL:
    pos[index] = rank[index].apply(lambda x: 1/amount if x<=amount else 0) 
#仓位时间调整：原本仓位信息在每周日，调整至每周四
pos.index -= pd.Timedelta(3,'d')
pos = pos.reindex(index=close.index).fillna(method='ffill')


#%% #计算收益
profit, cum_profit= close.copy(), dict()
for index in STOCK_POOL:
    profit[index] = (close[index]/close[index].shift(1) - 1) * pos[index]

cum_profit['single'] = 1 + (profit.T.sum()).cumsum() 
cum_profit['compound'] = (profit.T.sum()+1).cumprod() 

#%% 绘制最终收益率曲线
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