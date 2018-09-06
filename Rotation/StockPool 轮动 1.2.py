'''
读取本地股价数据
'''
import pandas as pd
import numpy as np
import talib

# STOCK_POOL = ['600000.SH', '600015.SH', '600029.SH', '600039.SZ', '600059.SZ', '600085.SH', '600132.SH']
STOCK_POOL = ['600048.SH', '600309.SH', '600585.SH', '000538.SZ', '000651.SZ', '600104.SH', '600519.SH', '601888.SH']
START_DATE = '2008-01-01'
END_DATE = '2018-08-16'
INITIAL_CAPITAL = 300
CAPITAL = INITIAL_CAPITAL/3

#%% 获取收盘数据
t1 = pd.read_excel(r"C:\Users\meiconte\Documents\RH\Historical Data\TRD_Dalyr_08-12.xlsx",
                   dtype={'date': 'datetime64', 'code': str})
t2 = pd.read_excel(r"C:\Users\meiconte\Documents\RH\Historical Data\TRD_Dalyr_13-18.xlsx",
                   dtype={'date': 'datetime64', 'code': str})
t3 = pd.read_excel(r"C:\Users\meiconte\Documents\RH\Historical Data\TRD_Dalyr_18.xlsx",
                   dtype={'date': 'datetime64', 'code': str})
t1.drop(columns=['open', 'high', 'low'], inplace=True)
t = t1.append(t2).append(t3)
# t = pd.read_excel(r"C:\Users\meiconte\Documents\RH\Historical Data\test.xlsx",dtype={'date': 'datetime64', 'code': str})
t.set_index('date', inplace=True)

tgroups = t.groupby('code')
price = {}
for symbol in STOCK_POOL:
    code = symbol[:-3]
    price[symbol] = tgroups.get_group(code).close
price = pd.DataFrame(price)
price.fillna(method='ffill', inplace=True)
print("Historical Price Loaded!")

#%% 计算相对强弱指数RSI
print("Calculating RSI")
RSI_Duration = 12
rsi = {}
for index in STOCK_POOL:
    rsi[index] = talib.RSI(price[index].shift(0), RSI_Duration)
rsi = pd.DataFrame(rsi)


#%% 分配仓位
# 总共买3个标的
amount = 3
# 关键：RSI越高越买，排序时需要用 ascending=False
rank = rsi.rank(axis=1, ascending=True)
price.fillna(value=0, inplace=True)  # 把早年暂未上市的股票价格填充为0
price_diff = price.diff()
price_pct_chg = price.pct_change()

# 赋权重(仓位)
pos = rsi.copy() 
for index in STOCK_POOL:
    pos[index] = rank[index].apply(lambda x: 1/amount if x <= amount else 0)
# 仓位时间调整
for day in pos.index:
    # print(day.dayofweek)
    if day.dayofweek != 4:
        pos.loc[day] = np.nan
        # pos.loc[day-pd.Timedelta(2+day.dayofweek,'D')]
pos.dropna(inplace=True)
pos = pos.reindex(index=price.index).fillna(method='ffill')
last_pos = pos.shift(1)


#%% 计算收益
'''
# 变量命名及含义
- capital_balance     账户余额
- share               持股份数
- confirmed_profit    确认收益
- accumulated_profit  累计收益（包含确认收益和浮动盈亏）
- profit_today        当日盈亏
- balance_today       当日账户余额（为账户余额的某一行）
- share_today         当日持股份数（为持股份数的某一行）
- win_count
- lose_count  
'''
# 添加首行（全为0）
first = {}
for symbol in STOCK_POOL:
    first[symbol] = 0
capital_balance, share = [first], [first]

confirmed_profit, profit_today = {}, {}  # 卖出后 已确认收益；继续持仓 浮动盈亏
win_count, lose_count = 0, 0
for day in price.index[RSI_Duration:]:
    balance_today, share_today = {'date': day}, {'date': day}
    for symbol in price.columns:
        if last_pos.loc[day, symbol] > pos.loc[day, symbol]:  # 信号提示要卖出
            print("卖出:", symbol, day.strftime('%Y-%m-%d'))
            share_today[symbol] = 0
            balance_today[symbol] = 0
            # 计算该笔交易最终收益 = 卖出时账户余额 - 买入时投入（CAPTIAL)
            profit = price.loc[day, symbol] * share[-1][symbol] - CAPITAL
            if profit >= 0:
                win_count += 1
            else:
                lose_count += 1
            confirmed_profit[day] = confirmed_profit.get(day, 0) + profit
            profit_today[day] = profit_today.get(day, 0) + price_diff.loc[day, symbol] * share[-1][symbol]
        elif last_pos.loc[day, symbol] < pos.loc[day, symbol]:  # 信号提示要买入
            print("信号提示要买入:", symbol, day.strftime('%Y-%m-%d'))
            share_today[symbol] = CAPITAL/price.loc[day, symbol]
            balance_today[symbol] = CAPITAL
        else:  # 继续持仓
            share_today[symbol] = share[-1][symbol]
            balance_today[symbol] = share[-1][symbol] * price.loc[day, symbol]
            profit_today[day] = profit_today.get(day, 0) + price_diff.loc[day, symbol] * share[-1][symbol]
    # 记录当日份数和账户余额
    share.append(share_today)
    capital_balance.append(balance_today)
# 转换数据类型
share = pd.DataFrame(share).set_index('date')
capital_balance = pd.DataFrame(capital_balance).set_index('date')


def dict_to_df(d: dict, columns_name: str):
    return pd.DataFrame(list(d.values()), index=d.keys(), columns=[columns_name])


profit_today = dict_to_df(profit_today, 'profit_today')
accumulated_profit = profit_today.cumsum().rename(
    columns={'profit_today': 'accumulated_profit'})

# confirmed_profit 原先是每天的确认收益，现在用.cumsum() 转成累计的
confirmed_profit = dict_to_df(confirmed_profit, 'confirmed_profit')
confirmed_profit = confirmed_profit.cumsum()

confirmed_profit.plot()
accumulated_profit.plot()

print("已完成交易次数：{}，其中：\n盈利交易{}次，亏损交易{}次;\n胜率{:.2%}".format(
    win_count+lose_count, win_count, lose_count, win_count/(win_count+lose_count)))

#%% 绘制最终收益率曲线
profit_summary = pd.concat(
    [confirmed_profit, accumulated_profit, profit_today], axis=1)
profit_summary.fillna(method='ffill',inplace=True)
# 最终版

import matplotlib.pyplot as plt

def generate_profit_curve(ans: pd.DataFrame, column='accumulated_profit'):
    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ans['NAV'] = ans[column]
    ans['Daily_pnl'] = ans[column].diff()

    ax = fig.add_subplot(211)
    ax.plot(ans.index, ans['NAV'], linewidth=2, label='累计收益')
    ax.plot(ans.index, ans['confirmed_profit'], linewidth=2, label='确认收益')
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
    if len(ans) > 30:
        width = 1.5
    bx.bar(ans.index, ans['Daily_pnl'].where(ans['Daily_pnl'] > 0),
           width, label='当日盈亏+', color='red', alpha=0.8)
    bx.bar(ans.index, ans['Daily_pnl'].where(ans['Daily_pnl'] < 0),
           width, label='当日盈亏-', color='green', alpha=0.8)
    bx.legend(fontsize=15)
    plt.grid()


generate_profit_curve(profit_summary)
plt.show()