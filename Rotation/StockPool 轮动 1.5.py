"""
Based on Version 1.4
--------------------
- 读取数据库价格
- 指数/股票均可读取
- 查看历史投入金额，位于money_invest变量中
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import time
from data_reader import get_muti_close_day, get_index_day, get_stock_day

TYPE = 0 # 0 for "index"; 1 for "stock"
STOCK_POOL = ['000016.SH','000905.SH','000009.SH','000991.SH','000935.SH','000036.SH']
# STOCK_POOL = ['600309.SH', '600585.SH', '000538.SZ', '000651.SZ', '600104.SH','601888.SH']
# STOCK_POOL = ["600104.SH"]
START_DATE = "2008-01-01"
END_DATE = "2018-11-16"
INITIAL_CAPITAL = 1000
# CAPITAL = INITIAL_CAPITAL / 3
DURATION = 250
Profit_Ceiling = [0.4, 0.2] #止盈线
Trailing_Percentage = 0.2 #优先止盈百分比

# %% 获取收盘数据
if TYPE==1:
    price = get_muti_close_day(STOCK_POOL,START_DATE,END_DATE)
# df = get_index_day('600519.SH',START_DATE,END_DATE)
elif TYPE==0:
    price = {}
    for symbol in STOCK_POOL:
        price[symbol] = get_index_day(symbol,START_DATE,END_DATE).sclose
    price = pd.DataFrame(price)
price.fillna(method="ffill", inplace=True)
print("Historical Price Loaded!")


# %% 技术指标 
price.fillna(value=0, inplace=True)  # 把早年暂未上市的股票价格填充为0
price_diff = price.diff()
price_pct_chg = price.pct_change()

price_std = price.rolling(window=DURATION).std()  # 计算标准差
R = (price / price.shift(1)).apply(np.log)
sigma = R.rolling(window=DURATION).std()
mu = (price / price.shift(DURATION)).apply(np.log) + 0.5 * np.square(sigma)

# %% 策略部分 分配仓位
# 买入时机
is_entry_time = np.square(sigma * 1) - mu > 0
percent_chg = (price / price.shift(int(DURATION/2))) - 1

# 赋权重(仓位)
pos = price.copy()
for index in STOCK_POOL:
    pos[index] = 1 / (1 + percent_chg[index] * 2 ) * 100
    pos[index][pos[index]<0]=0 #如果出现仓位<0，则补位为0
    pos *= is_entry_time
# 仓位时间调整
for day in pos.index:
    if day.dayofweek != 3:
        pos.loc[day] = np.nan
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
平均成本, 累计投入, 投入 = {}, {}, {}
for symbol in STOCK_POOL:
    first[symbol] = 0
    平均成本[symbol] = 10000
    累计投入[symbol] = 0
    投入[symbol] = 0
capital_balance, share = [first], [first]

confirmed_profit, profit_today = {}, {} # 卖出后 已确认收益；继续持仓 浮动盈亏
average_cost, investment, accumulated_investment = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
win_count, lose_count = 0, 0

def record_daily_profit(day, symbol):
    '''
    记录当日盈亏
    ----------
    用于没有交易产生时
    '''
    share_today[symbol] = share[-1][symbol]
    balance_today[symbol] = share[-1][symbol] * price.loc[day, symbol]
    profit_today[day] = profit_today.get(day, 0) + price_diff.loc[day, symbol] * share[-1][symbol]

def Sell(**arg):
    '''
    卖出函数
    ------
    传入参数：
    symbol, amount 或 percentage, price
    '''
    global 收回现金
    price = arg.pop('price', None)
    symbol = arg.pop('symbol', None)
    percentage = arg.pop('percentage', None)
    if percentage is None:
        amount = arg.pop('amount', None)
        if amount is None:
            raise ValueError("Please specify at least one argument, either \"amount\" or \"percentage\".")
        else:
            percentage = amount / share[-1][symbol]
            print("卖出 {:.1f} 份 {}".format(amount, symbol), day.strftime("%Y-%m-%d"))
    else:
        amount = share[-1][symbol] * percentage
        print(day.strftime("%Y-%m-%d"), "卖出 {:.2f} 份 {}".format(amount, symbol),"({:.1%} 持仓)".format(percentage), "卖价", price)
    
    share_today[symbol] = share[-1][symbol] - amount
    balance_today[symbol] = share_today[symbol] * price
    # 记录收益
    profit = (price - 平均成本[symbol]) * amount 
    confirmed_profit[day] = confirmed_profit.get(day, 0) + profit
    # print('confirmed_profit',day,confirmed_profit[day])
    #由于都是收盘时操作，所以计算当日盈亏应把卖出的份额也算上
    profit_today[day] = profit_today.get(day, 0) + price_diff.loc[day, symbol] * share[-1][symbol] 
    # 如果全额卖出，平均成本会变为 0/0，因此对此情况做预先处理
    if share_today[symbol] == 0:
        平均成本[symbol] = 10000
    else:
        平均成本[symbol] = (平均成本[symbol] * share[-1][symbol] - amount * price)/ share_today[symbol]
    投入[symbol] *= (1-percentage)
    收回现金 += amount * price
    #print("收回现金", 收回现金)
    return "卖出成功"

def Buy(**arg):
    '''
    买入函数
    ------
    传入参数：
    symbol, amount 或 percentage, price
    '''
    price = arg.pop('price', None)
    symbol = arg.pop('symbol', None)
    
    percentage = arg.pop('percentage', None)
    if percentage is None:
        amount = arg.pop('amount', None)
        if amount is None:
            raise ValueError("Please specify at least one argument, either \"amount\" or \"percentage\".")
    else:
        amount = share[-1][symbol] * percentage
    share_today[symbol] = share[-1][symbol] + amount
    balance_today[symbol] = capital_balance[-1][symbol]+ amount * price
    
    # 终于找到bug了！忘记在买入的那天记录当日盈亏！
    profit_today[day] = profit_today.get(day, 0) + price_diff.loc[day, symbol] * share[-1][symbol] 

    # 平均成本 = (原平均成本 * 原份数 + 新买入份数 * 买入价格) / 总份数
    平均成本[symbol] = (平均成本[symbol] * share[-1][symbol] + amount * price) / share_today[symbol]
    
    # 累计投入 = 原累计投入 + 新买入份数 * 买入价格
    投入[symbol] += amount * price
    累计投入[symbol] += amount * price
    
    print(day.strftime("%Y-%m-%d"), "买入 {:.1f}份 {}, 当前平均成本 {:.2f}".format(amount, symbol, 平均成本[symbol]), end='\r')
    return "买入成功"
def Check_Signal():
    '''
    检查交易信号  
    ----------

    - 此部分为策略核心部分。回测时，每个Bar会自动检查是否存在交易信号。
    - BUY_SIGNALS, SELL_SIGNALS 为 dict 类型  
      必须包含的键：{'symbol', 'percentage'或'amount', 'price'}
    - 一般 BUY_SIGNALS 的结构为：
      {"600048.SH":{"定投买入":{'symbol':"600048.SH"},
                    "卖出":{'symbol':"600048.SH"}},
       "601888.SH":{"定投买入":{'symbol':"601888.SH"},
                    "卖出":{'symbol':"601888.SH"}}
      }
    '''
    BUY_SIGNALS, SELL_SIGNALS = dict(), dict()
    total_share_required = 0
    for symbol in STOCK_POOL:
        SELL_SIGNALS[symbol] = {}
        BUY_SIGNALS[symbol] = {}
        if price.loc[day, symbol] > 平均成本[symbol] * (1 + Profit_Ceiling[0]) and share[-1][symbol] > 0:
            SELL_SIGNALS[symbol]["全部止盈"] = {'symbol':symbol, 'percentage':1, 'price':price.loc[day, symbol]}
        elif price.loc[day, symbol] > 平均成本[symbol] * (1 + Profit_Ceiling[1]) and share[-1][symbol] > 500:
            SELL_SIGNALS[symbol]["部分止盈"] = {'symbol':symbol, 'percentage':Trailing_Percentage, 'price':price.loc[day, symbol]}
        elif pos.loc[day, symbol] > 0:
            BUY_SIGNALS[symbol]["定投买入"] = {'symbol':symbol, 'amount':pos.loc[day, symbol], 'price':price.loc[day, symbol]}
            total_share_required += pos.loc[day, symbol] 
    for symbol in STOCK_POOL:
        try:
            BUY_SIGNALS[symbol]["定投买入"]['amount'] *= INITIAL_CAPITAL/total_share_required / price.loc[day, symbol]
            if BUY_SIGNALS[symbol]["定投买入"]['amount']<=0:BUY_SIGNALS[symbol]["定投买入"]['amount']=0.01
        except:
            pass
            
    return BUY_SIGNALS, SELL_SIGNALS

# 按天回测正式开始
t0 = time.time()
收回现金 = 0
for day in price.index[DURATION:]:
    balance_today, share_today = {"date": day}, {"date": day}
    buy_signals, sell_signals = Check_Signal()
    
    # print(buy_signals, sell_signals)
    for symbol in STOCK_POOL: 
        for signal in sell_signals[symbol]:
            Sell(**sell_signals[symbol][signal])
        for signal in buy_signals[symbol]:
            Buy(**buy_signals[symbol][signal])
        if len(buy_signals.get(symbol,{}))==0 and len(sell_signals.get(symbol,{}))==0:
            record_daily_profit(day, symbol)
        average_cost.loc[day, symbol] = 平均成本[symbol] if 平均成本[symbol] < 10000 else np.nan
        investment.loc[day, symbol] = 投入[symbol]
        accumulated_investment.loc[day, symbol] = 累计投入[symbol]
    # 记录当日份数和账户余额
    share.append(share_today)
    capital_balance.append(balance_today)
t1 = time.time()
tpy = t1 - t0
print('\n回测已完成，用时 %5.3f 秒' % tpy)
# 转换数据类型
share = pd.DataFrame(share).set_index("date")#.rename(columns={'600519.SH':'shares'})
#investment = investment.rename(columns={'600519.SH':'资金投入'})
capital_balance = pd.DataFrame(capital_balance).set_index("date")


def dict_to_df(d: dict, columns_name: str):
    return pd.DataFrame(list(d.values()), index=d.keys(), columns=[columns_name])

profit_today = dict_to_df(profit_today, "profit_today")
accumulated_profit = profit_today.cumsum().rename(columns={"profit_today": "accumulated_profit"})

# confirmed_profit 原先是每天的确认收益，现在用.cumsum() 转成累计的
confirmed_profit = dict_to_df(confirmed_profit, "confirmed_profit").cumsum()
#average_cost = dict_to_df(average_cost, "average_cost")
#investment = dict_to_df(investment, "investment")

#average_cost = average_cost.rename(columns={'600519.SH':"average_cost"})
average_cost.plot(title="持股平均成本")
investment.plot(title="投入资金")

print("\n----- 策略回测报告 -----")
print("标的", STOCK_POOL)
print("累计投入\t{:.2f}".format(accumulated_investment.iloc[-1].sum())) # investment.T.sum().max()
print("当前持仓总市值\t{:.2f}".format(capital_balance.iloc[-1].sum()+收回现金))#confirmed_profit.iloc[-1,0]
print("投入资金平均值\t￥{:.2f}".format(investment.T.sum().mean()))
rate_of_return = accumulated_profit.iloc[-1,0]/investment.T.sum().mean()
print("总收益率\t{:.2%}".format(rate_of_return))
print("止盈参数\t{:.2%}".format(Profit_Ceiling[0]))
print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(accumulated_investment.iloc[-1].sum(),capital_balance.iloc[-1].sum()+收回现金,investment.T.sum().mean(),rate_of_return))
# print("平均年化收益\t",)
# print(
#     "已完成交易次数：{}，其中：\n盈利交易{}次，亏损交易{}次;\n胜率{:.2%}".format(
#         win_count + lose_count,
#         win_count,
#         lose_count,
#         win_count / (win_count + lose_count),
#     )
# )


# %% 绘制最终收益率曲线
profit_summary = pd.concat([price,investment,share,average_cost,confirmed_profit, accumulated_profit, profit_today], axis=1)
profit_summary.fillna(method="ffill", inplace=True)
# 最终版

def generate_profit_curve(ans: pd.DataFrame, column="accumulated_profit"):
    import matplotlib.pyplot as plt

    print("正在生成图像...")
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
plt.show()

#%% 最新指导
print("最新定投金额指导：")
money_invest = (pos.T/pos.T.sum()*1000).T
latest_guidance = money_invest.drop_duplicates().iloc[-1]

print(latest_guidance.round(0))
