"""
读取本地股价数据
改写为定投
"""
import pandas as pd
import numpy as np
import talib

# STOCK_POOL = ['600000.SH', '600015.SH', '600029.SH', '600039.SZ', '600059.SZ', '600085.SH', '600132.SH']
# STOCK_POOL = ['600048.SH', '600309.SH', '600585.SH', '000538.SZ', '000651.SZ', '600104.SH', '600519.SH', '601888.SH']
STOCK_POOL = ["000651.SH"]
START_DATE = "2008-01-01"
END_DATE = "2018-08-16"
INITIAL_CAPITAL = 300
CAPITAL = INITIAL_CAPITAL / 3
DURATION = 10 * 5
Profit_Ceiling = [0.3, 0.2] #止盈线
Trailing_Percentage = 0.2 #优先止盈百分比

# %% 获取收盘数据
t1 = pd.read_excel(
    r"C:\Users\meiconte\Documents\RH\Historical Data\TRD_Dalyr_08-12.xlsx",
    dtype={"date": "datetime64", "code": str},
)
t2 = pd.read_excel(
    r"C:\Users\meiconte\Documents\RH\Historical Data\TRD_Dalyr_13-18.xlsx",
    dtype={"date": "datetime64", "code": str},
)
t3 = pd.read_excel(
    r"C:\Users\meiconte\Documents\RH\Historical Data\TRD_Dalyr_18.xlsx",
    dtype={"date": "datetime64", "code": str},
)
t1.drop(columns=["open", "high", "low"], inplace=True)
t = t1.append(t2).append(t3)
# t = pd.read_excel(r"C:\Users\meiconte\Documents\RH\Historical Data\test.xlsx",dtype={'date': 'datetime64', 'code': str})
t.set_index("date", inplace=True)

tgroups = t.groupby("code")
price = {}
for symbol in STOCK_POOL:
    code = symbol[:-3]
    price[symbol] = tgroups.get_group(code).close
price = pd.DataFrame(price)
price.fillna(method="ffill", inplace=True)
print("Historical Price Loaded!")


# %% 技术指标 计算相对强弱指数RSI
# print("Calculating RSI")
# RSI_DURATION = 12
# rsi = {}
# for index in STOCK_POOL:
#     rsi[index] = talib.RSI(price[index].shift(-1), RSI_DURATION)
# rsi = pd.DataFrame(rsi)
price.fillna(value=0, inplace=True)  # 把早年暂未上市的股票价格填充为0
price_diff = price.diff()
price_pct_chg = price.pct_change()

price_std = price.rolling(window=DURATION).std()  # 计算标准差
R = (price / price.shift(1)).apply(np.log)
sigma = R.rolling(window=DURATION).std()
mu = (price / price.shift(DURATION)).apply(np.log) + 0.5 * np.sqrt(sigma)

# %% 策略部分 分配仓位
# 买入时机
is_entry_time = np.square(sigma * 3) - mu > -0.3
percent_chg = (price / price.shift(int(DURATION/2))) - 1

# 赋权重(仓位)
pos = price.copy()
for index in STOCK_POOL:
    pos[index] = 1 / (1 + percent_chg[index] * 2 ) * 100
    #pos[index][pos[index]<0]=0 #如果出现仓位<0，则补位为0
    pos *= is_entry_time
# 仓位时间调整
for day in pos.index:
    # print(day.dayofweek)
    if day.dayofweek != 4:
        pos.loc[day] = np.nan
        # pos.loc[day-pd.Timedelta(2+day.dayofweek,'D')]
#pos.dropna(inplace=True)
pos = pos.reindex(index=price.index).fillna(0)
#此时pos实际上是买入的信号，而不是实时仓位


# %% 回测模块 计算收益
"""
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
- average_cost        平均持仓成本
- investment          累计投入
"""
# 添加首行（全为0）
first = {}
平均成本, 累计投入 = {}, {}
for symbol in STOCK_POOL:
    first[symbol] = 0
    平均成本[symbol] = 10000
    累计投入[symbol] = 0
capital_balance, share = [first], [first]

confirmed_profit, profit_today, average_cost, investment = {}, {}, {}, {}  # 卖出后 已确认收益；继续持仓 浮动盈亏
win_count, lose_count = 0, 0

def record_daily_profit(day, symbol):
    share_today[symbol] = share[-1][symbol]
    balance_today[symbol] = share[-1][symbol] * price.loc[day, symbol]
    profit_today[day] = profit_today.get(day, 0) + price_diff.loc[day, symbol] * share[-1][symbol]

已经部分止盈 = False
for day in price.index[DURATION:]:
    balance_today, share_today = {"date": day}, {"date": day}
    
    for symbol in price.columns:
        if price.loc[day, symbol] > 平均成本[symbol] * (1 + Profit_Ceiling[0]):  # 信号提示要卖出
            if share[-1][symbol] > 0:
                print("全部止盈卖出:", symbol, day.strftime("%Y-%m-%d"))
                share_today[symbol] = 0
                balance_today[symbol] = 0
                已经部分止盈 = False
                profit = (price.loc[day, symbol] - 平均成本[symbol]) * share[-1][symbol] 
                confirmed_profit[day] = confirmed_profit.get(day, 0) + profit
                profit_today[day] = profit_today.get(day, 0) + price_diff.loc[day, symbol] * share[-1][symbol]
                平均成本[symbol] = 10000
                累计投入[symbol] = 0
            else:
                record_daily_profit(day, symbol)
        elif price.loc[day, symbol] > 平均成本[symbol] * (1 + Profit_Ceiling[1]):
            if share[-1][symbol] > 100 and not 已经部分止盈:
                print("部分止盈卖出:", symbol, day.strftime("%Y-%m-%d"))
                share_today[symbol] = share[-1][symbol] * (1 - Trailing_Percentage)
                balance_today[symbol] = share_today[symbol] * price.loc[day, symbol]
                已经部分止盈 = True
                profit = (price.loc[day, symbol] - 平均成本[symbol]) * share[-1][symbol] * Trailing_Percentage
                confirmed_profit[day] = confirmed_profit.get(day, 0) + profit
                profit_today[day] = profit_today.get(day, 0) + price_diff.loc[day, symbol] * share[-1][symbol]
                平均成本[symbol] = (平均成本[symbol] * share[-1][symbol] - share[-1][symbol] * Trailing_Percentage * price.loc[day, symbol])/ share_today[symbol]
                累计投入[symbol] *= (1-Trailing_Percentage)
            else:
                record_daily_profit(day, symbol)
        elif pos.loc[day, symbol]>0:  # 信号提示要买入
            print("定投买入:", symbol, day.strftime("%Y-%m-%d"), 平均成本[symbol])
            share_today[symbol] = share[-1][symbol] + pos.loc[day, symbol]
            balance_today[symbol] = capital_balance[-1][symbol] + pos.loc[day, symbol] * price.loc[day, symbol]
            # 平均成本 = (原平均成本 * 原份数 + 新买入份数 * 买入价格) / 总份数
            平均成本[symbol] = (平均成本[symbol] * share[-1][symbol] + pos.loc[day, symbol] * price.loc[day, symbol]) / share_today[symbol]
            累计投入[symbol] += pos.loc[day, symbol] * price.loc[day, symbol]
        else:  # 继续持仓
            record_daily_profit(day, symbol)
    # 记录当日份数和账户余额
    share.append(share_today)
    capital_balance.append(balance_today)
    average_cost[day] = 平均成本[symbol] if 平均成本[symbol] < 10000 else np.nan
    investment[day] = 累计投入[symbol]
# 转换数据类型
share = pd.DataFrame(share).set_index("date")
capital_balance = pd.DataFrame(capital_balance).set_index("date")


def dict_to_df(d: dict, columns_name: str):
    return pd.DataFrame(list(d.values()), index=d.keys(), columns=[columns_name])


profit_today = dict_to_df(profit_today, "profit_today")
accumulated_profit = profit_today.cumsum().rename(
    columns={"profit_today": "accumulated_profit"}
)

# confirmed_profit 原先是每天的确认收益，现在用.cumsum() 转成累计的
confirmed_profit = dict_to_df(confirmed_profit, "confirmed_profit")
confirmed_profit = confirmed_profit.cumsum()

average_cost = dict_to_df(average_cost, "average_cost")
investment = dict_to_df(investment, "investment")

confirmed_profit.plot()
accumulated_profit.plot()
average_cost.plot()
investment.plot()
# print(
#     "已完成交易次数：{}，其中：\n盈利交易{}次，亏损交易{}次;\n胜率{:.2%}".format(
#         win_count + lose_count,
#         win_count,
#         lose_count,
#         win_count / (win_count + lose_count),
#     )
# )


# %% 绘制最终收益率曲线
profit_summary = pd.concat([confirmed_profit, accumulated_profit, profit_today], axis=1)
profit_summary.fillna(method="ffill", inplace=True)
# 最终版


def generate_profit_curve(ans: pd.DataFrame, column="accumulated_profit"):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ans["NAV"] = ans[column]
    ans["Daily_pnl"] = ans[column].diff()

    ax = fig.add_subplot(211)
    ax.plot(ans.index, ans["NAV"], linewidth=2, label="累计收益")
    ax.plot(ans.index, ans["confirmed_profit"], linewidth=2, label="确认收益")
    ax.fill_between(
        ans.index,
        ans["NAV"],
        y2=1,
        where=(ans["NAV"] < ans["NAV"].shift(1))
        | ((ans["NAV"] > ans["NAV"].shift(-1)) & (ans["NAV"] >= ans["NAV"].shift(1))),
        facecolor="grey",
        alpha=0.3,
    )
    # 最大回撤标注
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    ax.legend(fontsize=15)
    plt.grid()

    bx = fig.add_subplot(212)
    width = 1
    if len(ans) > 30:
        width = 1.5
    bx.bar(
        ans.index,
        ans["Daily_pnl"].where(ans["Daily_pnl"] > 0),
        width,
        label="当日盈亏+",
        color="red",
        alpha=0.8,
    )
    bx.bar(
        ans.index,
        ans["Daily_pnl"].where(ans["Daily_pnl"] < 0),
        width,
        label="当日盈亏-",
        color="green",
        alpha=0.8,
    )
    bx.legend(fontsize=15)
    plt.grid()


generate_profit_curve(profit_summary)
