# import sys 
# sys.path.append(r"C:\Users\meiconte\Documents\RH\Summary")

import pandas as pd
import numpy as np
from data_reader import get_stock_day   
import talib

STOCK_POOL = ['600048.SH','600309.SH','600585.SH','000538.SZ','000651.SZ','600104.SH','600519.SH','601888.SH']
START_DATE = '2008-01-01'
END_DATE = '2018-08-16'
INITIAL_CAPITAL = 300
CAPITAL = INITIAL_CAPITAL/3
# FILEPATH = r"C:\Users\meiconte\Documents\RH\excel\TSoutput_stock\\"

# worths, stock_close = {}, {}
# for index in STOCK_POOL:
#     print("----- Now processing {} ...".format(index))
#     s.read_files(FILEPATH + index + '.xlsx')
#     s.start()
#     s.AIP_net()
#     s.generate_profit_curve()
#     print("----- Processed successfully! -----", end='\n\n')
#     worths[index]=s.ans
#     stock_close[index]=s.stock_close.sclose
# start_date = worths[index].index[0]

#%% 获取收盘数据
price = {}
for symbol in STOCK_POOL:
    temp = get_stock_day(symbol, START_DATE, END_DATE, '1D')
    price[symbol] = temp.sclose
price = pd.DataFrame(price)
price.fillna(method='ffill', inplace = True)
price_diff = price.diff()
price_pct_chg = price.pct_change()

#%% 计算相对强弱指数RSI
RSI_Duration = 12
rsi={}
for index in STOCK_POOL:
    rsi[index] = talib.RSI(price[index].shift(-1), RSI_Duration)
rsi = pd.DataFrame(rsi)


#%% 分配仓位
# 总共买3个标的
amount = 3
# 关键：RSI越高越买，排序时需要用 ascending=False
rank=rsi.rank(axis=1, ascending=False)

# 赋权重(仓位)
pos=rsi.copy()
for index in STOCK_POOL:
    pos[index] = rank[index].apply(lambda x: 1/amount if x<=amount else 0) 
#仓位时间调整
for day in pos.index:
    print(day.dayofweek)
    if day.dayofweek != 4:
        pos.loc[day] = np.nan
        # pos.loc[day-pd.Timedelta(2+day.dayofweek,'D')]
pos.dropna(inplace=True)
pos = pos.reindex(index = price.index).fillna(method='ffill')
last_pos=pos.shift(1)


#%% 计算收益
# 添加首行
first={}
for symbol in STOCK_POOL:
    first[symbol] = 0 
capital_balance, share = [first], [first]

confirmed_profit, profit_today = {}, {} # 卖出后 已确认收益；继续持仓 浮动盈亏
for day in price.index[RSI_Duration:]:
    balance_today, share_today = {'date':day}, {'date':day}
    for symbol in price.columns:
        if last_pos.loc[day, symbol]>pos.loc[day, symbol]: # 信号提示要卖出
            print("卖出:", symbol, day)
            share_today[symbol] = 0
            balance_today[symbol] = 0
            confirmed_profit[day] = confirmed_profit.get(day, 0) + price.loc[day, symbol] * share[-1][symbol] - CAPITAL
        elif last_pos.loc[day, symbol]<pos.loc[day, symbol]: # 信号提示要买入
            print("信号提示要买入:", symbol, day)
            share_today[symbol] = CAPITAL/price.loc[day, symbol]
            balance_today[symbol] = CAPITAL
        else: # 继续持仓
            share_today[symbol] = share[-1][symbol]
            profit_today[day] = profit_today.get(day, 0) + price_diff.loc[day, symbol] * share[-1][symbol]
            balance_today[symbol] = share[-1][symbol] * price.loc[day, symbol]

    share.append(share_today)
    capital_balance.append(balance_today)

share=pd.DataFrame(share).set_index('date')
capital_balance=pd.DataFrame(capital_balance).set_index('date')
def dict_to_df(d:dict):
    return pd.DataFrame(list(d.values()), index = d.keys())
confirmed_profit = dict_to_df(confirmed_profit)
confirmed_profit = confirmed_profit.cumsum()
profit_today = dict_to_df(profit_today)
profit_today = profit_today.cumsum()

confirmed_profit.plot()


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