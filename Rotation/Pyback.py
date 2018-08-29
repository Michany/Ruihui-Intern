__doc__ = """
pyback - an experimental module to perform backtests for Python
=====================================================================

Main Features
-------------
Here are something this backtest framework can do:

- 读取数据库价格；
  也可读取给定的价格DataFrame，并计算刷新各个技术指标；
  指数/股票均可读取
- TODO 盈利再投资
- 分级止盈赎回，可以分任意多个等级

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import time
from collections import Iterable
from data_reader import get_muti_close_day, get_index_day, get_stock_day


class Backtest:
    """
    Wrapper class for a backtest.
    """

    def __init__(self, stock_pool, start_date="2008-01-01", end_date="2018-08-24", 
                 **arg):
        self.START_DATE = start_date
        self.END_DATE = end_date
        self.STOCK_POOL = stock_pool

        initial_capital = arg.pop("initial_capital", None)
        if initial_capital is None:
            self.INITIAL_CAPITAL = 1000
        elif initial_capital <= 0:
            raise ValueError("Initial capital must be greater than zero.")
        else:
            self.INITIAL_CAPITAL = initial_capital

        TYPE = arg.pop("type", None)
        if TYPE == 1 or TYPE == "stock":
            self.TYPE = 1
        elif TYPE == 0 or TYPE == "index":
            self.TYPE = 0
        else:
            raise ValueError("Type not recognized.")

        self.DURATION = arg.pop("duration", 250)
        self.Profit_Ceiling = arg.pop("profit_ceiling", [0.4, 0.2])  # 止盈线
        self.Trailing_Percentage = arg.pop("trailing_percentage", [1, 0.2])    # 优先止盈百分比

    def _init_data(self, type):
        """
        Fetch Data
        ----------
        获取收盘数据
        
        Type 0 for "index"; 1 for "stock"
        """
        # price = get_muti_close_day(STOCK_POOL,START_DATE,END_DATE)
        # df = get_index_day('600519.SH',START_DATE,END_DATE)
        if type == 1:
            price = get_muti_close_day(self.STOCK_POOL, self.START_DATE, self.END_DATE)
        # df = get_index_day('600519.SH',START_DATE,END_DATE)
        elif type == 0:
            price = {}
            for symbol in self.STOCK_POOL:
                price[symbol] = get_index_day(symbol, self.START_DATE, self.END_DATE).sclose
            price = pd.DataFrame(price)
        price.fillna(method="ffill", inplace=True)
        print("Historical Price Loaded!")
        
        self.price = price
        del price
    
    @property
    def Profit_Ceiling(self):
        return self._profit_ceiling
    @Profit_Ceiling.setter
    def Profit_Ceiling(self, profit_ceiling_parameters):
        if isinstance(profit_ceiling_parameters, Iterable):
            try:
                self._profit_ceiling = list(float(i) for i in profit_ceiling_parameters)
            except:
                raise ValueError("Profit Ceiling must be a number Type")
        elif isinstance(profit_ceiling_parameters, (float,int)):
            self._profit_ceiling = [profit_ceiling_parameters]
        else:
            raise ValueError("Profit Ceiling must be a Python builtin number Type")
    
    @property
    def Trailing_Percentage(self):
        return self._trailing_percentage
    @Trailing_Percentage.setter
    def Trailing_Percentage(self, trailing_percentage_parameters):
        self._trailing_percentage = trailing_percentage_parameters

    @property
    def StopLoss(self):
        return self._stoploss
    @StopLoss.setter
    def StopLoss(self, stoploss_parameters):
        self._stoploss = stoploss_parameters

    @property
    def price(self):
        return self._price 
    @price.setter
    def price(self, price_DataFrame):
        if isinstance(price_DataFrame, pd.DataFrame):
            self._price = price_DataFrame
            self.technical_indicators() #给price赋值时，重新计算技术指标
            self._pos = self.timing()
        else:
            raise ValueError("Unable to handle the inputted price data of type {}, must be <pd.DataFrame>.".format(type(price_DataFrame)))
    @property
    def pos(self):
        return self._pos
    @pos.setter
    def pos(self, pos_DataFrame):
        if isinstance(pos_DataFrame, pd.DataFrame):
            self._pos = pos_DataFrame
        else:
            raise ValueError("Unable to handle the inputted position data of type {}, must be <pd.DataFrame>.".format(type(pos_DataFrame)))            

    # %% 技术指标
    def technical_indicators(self):
        self._price.fillna(value=0, inplace=True)  # 把早年暂未上市的股票价格填充为0
        self._price_diff = self._price.diff()
        self._price_pct_chg = self._price.pct_change()

        self._price_std = self._price.rolling(window=self.DURATION).std()  # 计算标准差
        self._R = (self._price / self._price.shift(1)).apply(np.log)
        self._sigma = self._R.rolling(window=self.DURATION).std()
        self._mu = (self._price / self._price.shift(self.DURATION)).apply(np.log) + 0.5 * np.sqrt(self._sigma)

    def timing(self):
        # %% 策略部分 分配仓位
        # 买入时机
        is_entry_time = np.square(self._sigma * 1) - self._mu > -0.3
        percent_chg = (self.price / self.price.shift(int(self.DURATION / 2))) - 1

        # 赋权重(仓位)
        pos = self.price.copy()
        for index in self.STOCK_POOL:
            pos[index] = 1 / (1 + percent_chg[index] * 2) * 100
            pos[index][pos[index] < 0] = 0  # 如果出现仓位<0，则补位为0
        pos *= is_entry_time

        # 仓位时间调整
        for day in pos.index:
            if day.dayofweek != 4:
                pos.loc[day] = np.nan
        pos = pos.reindex(index=self.price.index).fillna(0)
        # 此时pos实际上是买入的信号，而不是实时仓位
        print("Timing calculated")
        return pos

    def loop(self):
        """
        Loop is the main method for <Class Backtest>.
        ============================================

        变量命名及含义
        -----------
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
        def _init_loop():
            # 添加首行（全为0）
            first = {}
            self.平均成本, self.累计投入, self.投入 = {}, {}, {}
            for symbol in self.STOCK_POOL:
                first[symbol] = 0
                self.平均成本[symbol] = 10000
                self.累计投入[symbol] = 0
                self.投入[symbol] = 0
            self.capital_balance, self.share = [first], [first]

            self.confirmed_profit, self.profit_today = {}, {}  # 卖出后 已确认收益；继续持仓 浮动盈亏
            self.average_cost, self.investment, self.accumulated_investment = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            self.win_count, self.lose_count = 0, 0

        def record_daily_profit(day, symbol):
            """
            记录当日盈亏
            ----------
            用于没有交易产生时
            """
            self.share_today[symbol] = self.share[-1][symbol]
            self.balance_today[symbol] = self.share[-1][symbol] * self.price.loc[day, symbol]
            self.profit_today[day] = (self.profit_today.get(day, 0) + self._price_diff.loc[day, symbol] * self.share[-1][symbol])

        def Sell(**arg):
            """
            卖出函数
            ------
            传入参数：
            symbol, amount 或 percentage, price
            """
            price = arg.pop("price", None)
            symbol = arg.pop("symbol", None)
            percentage = arg.pop("percentage", None)
            if percentage is None:
                amount = arg.pop("amount", None)
                if amount is None:
                    raise ValueError('Please specify at least one argument, either "amount" or "percentage".')
                else:
                    percentage = amount / self.share[-1][symbol]
                    print("卖出 {:.1f} 份 {}".format(amount, symbol), day.strftime("%Y-%m-%d"),)
            else:
                amount = self.share[-1][symbol] * percentage
                print(day.strftime("%Y-%m-%d"), "卖出 {:.2f} 份 {}".format(amount, symbol),
                    "({:.1%} 持仓)".format(percentage), "卖价", price)

            self.share_today[symbol] = self.share[-1][symbol] - amount
            self.balance_today[symbol] = self.share_today[symbol] * price
            # 记录收益
            profit = (price - self.平均成本[symbol]) * amount
            self.confirmed_profit[day] = self.confirmed_profit.get(day, 0) + profit
            # print('confirmed_profit',day,confirmed_profit[day])
            # 由于都是收盘时操作，所以计算当日盈亏应把卖出的份额也算上
            self.profit_today[day] = (self.profit_today.get(day, 0) + self._price_diff.loc[day, symbol] * self.share[-1][symbol])
            # 如果全额卖出，平均成本会变为 0/0，因此对此情况做预先处理
            if self.share_today[symbol] == 0:
                self.平均成本[symbol] = 10000
            else:
                # TODO 卖出时到底应不应该调整成本？
                pass#self.平均成本[symbol] = (self.平均成本[symbol] * self.share[-1][symbol] - amount * price) / self.share_today[symbol]
            self.投入[symbol] *= 1 - percentage
            self.cash_balance += amount * price
            return "卖出成功"

        def Buy(**arg):
            """
            买入函数
            ------
            传入参数：
            symbol, amount 或 percentage, price
            """
            price = arg.pop("price", None)
            symbol = arg.pop("symbol", None)

            percentage = arg.pop("percentage", None)
            if percentage is None:
                amount = arg.pop("amount", None)
                if amount is None:
                    raise ValueError('Please specify at least one argument, either "amount" or "percentage".')
                else:
                    percentage = amount / self.share[-1][symbol]
            else:
                amount = self.share[-1][symbol] * percentage
            self.share_today[symbol] = self.share[-1][symbol] + amount
            self.balance_today[symbol] = self.capital_balance[-1][symbol] + amount * price

            # 终于找到bug了！忘记在买入的那天记录当日盈亏！
            self.profit_today[day] = (self.profit_today.get(day, 0) + self._price_diff.loc[day, symbol] * self.share[-1][symbol])

            # 平均成本 = (原平均成本 * 原份数 + 新买入份数 * 买入价格) / 总份数
            self.平均成本[symbol] = (self.平均成本[symbol] * self.share[-1][symbol] + amount * price) / self.share_today[symbol]

            # 累计投入 = 原累计投入 + 新买入份数 * 买入价格
            self.累计投入[symbol] += amount * price
            self.投入[symbol] += amount * price

            print(
                day.strftime("%Y-%m-%d"),
                "买入 {:.1f}份 {}, 当前平均成本 {:.2f}".format(amount, symbol, self.平均成本[symbol]),
                end="\r",
            )
            return "买入成功"

        def Check_Signal(price_today):
            """
            检查交易信号  
            ----------
            此部分为策略核心部分。回测时，每个Bar会自动检查是否存在交易信号。

            - BUY_SIGNALS, SELL_SIGNALS 为 dict 类型  
            必须包含的键：{'symbol', 'percentage'或'amount', 'price'}
            - 一般 BUY_SIGNALS 的结构为：
            >>> {"600048.SH":{"定投买入":{'symbol':"600048.SH"}, "卖出":{'symbol':"600048.SH"}},
                 "601888.SH":{"定投买入":{'symbol':"601888.SH"}, "卖出":{'symbol':"601888.SH"}}
            }
            """
            BUY_SIGNALS, SELL_SIGNALS = dict(), dict()
            total_share_required = 0
            buybuybuy=False
            for symbol in self.STOCK_POOL:
                SELL_SIGNALS[symbol], BUY_SIGNALS[symbol] = {}, {}

                #止盈
                for i in range(len(self.Profit_Ceiling)):
                    if (price_today[symbol] > self.平均成本[symbol] * (1 + self.Profit_Ceiling[i]) and self.share[-1][symbol] > 0):
                        SELL_SIGNALS[symbol]["{:.0%}止盈".format(self.Profit_Ceiling[i])] = {
                            "symbol": symbol,
                            "percentage": self.Trailing_Percentage[i],
                            "price": price_today[symbol],
                        }
                        # break语句 保证每次只有一种止盈
                        break
                
                #止损

                #买入
                self.盈利再投资 = self.cash_balance/100
                if len(SELL_SIGNALS) != 0:
                    if self.pos.loc[day, symbol] > 0:
                        BUY_SIGNALS[symbol]["定投买入"] = {
                            "symbol": symbol,
                            "amount": self.pos.loc[day, symbol],
                            "price": price_today[symbol],
                        }
                        total_share_required += self.pos.loc[day, symbol]
                        buybuybuy = True
            for symbol in self.STOCK_POOL:
                try:
                    BUY_SIGNALS[symbol]["定投买入"]["amount"] *= (
                        (self.INITIAL_CAPITAL + self.盈利再投资)/ total_share_required / price_today[symbol])
                except:
                    pass
            if buybuybuy:self.cash_balance -= self.盈利再投资
            return BUY_SIGNALS, SELL_SIGNALS

        self._init_data(type=self.TYPE)
        _init_loop()
        # 按天回测正式开始
        t0 = time.time()
        self.cash_balance = 0
        for day in self.price.index[self.DURATION:]:
            self.balance_today, self.share_today = {"date": day}, {"date": day}
            buy_signals, sell_signals = Check_Signal(self.price.loc[day])
            # print(buy_signals, sell_signals)
            for symbol in self.STOCK_POOL:
                for signal in sell_signals[symbol]:
                    Sell(**sell_signals[symbol][signal])
                for signal in buy_signals[symbol]:
                    Buy(**buy_signals[symbol][signal])
                if (len(buy_signals.get(symbol, {})) == 0 and len(sell_signals.get(symbol, {})) == 0):
                    record_daily_profit(day, symbol)
                self.average_cost.loc[day, symbol] = (self.平均成本[symbol] if self.平均成本[symbol] < 10000 else np.nan)
                self.investment.loc[day, symbol] = self.投入[symbol]
                self.accumulated_investment.loc[day, symbol] = self.累计投入[symbol]
            # 记录当日份数和账户余额
            self.share.append(self.share_today)
            self.capital_balance.append(self.balance_today)
        t1 = time.time()
        tpy = t1 - t0
        print("回测已完成，用时 %5.3f 秒" % tpy)

        def convert_types():
            # 转换数据类型
            self.share = pd.DataFrame(self.share).set_index("date")
            # investment = investment.rename(columns={'600519.SH':'资金投入'})
            self.capital_balance = pd.DataFrame(self.capital_balance).set_index("date")

            def dict_to_df(d: dict, columns_name: str):
                return pd.DataFrame(
                    list(d.values()), index=d.keys(), columns=[columns_name]
                )

            self.profit_today = dict_to_df(self.profit_today, "profit_today")
            self.accumulated_profit = self.profit_today.cumsum().rename(
                columns={"profit_today": "accumulated_profit"}
            )

            # confirmed_profit 原先是每天的确认收益，现在用.cumsum() 转成累计的
            self.confirmed_profit = dict_to_df(self.confirmed_profit, "confirmed_profit").cumsum()

        convert_types()
        # average_cost = average_cost.rename(columns={'600519.SH':"average_cost"})
        self.average_cost.plot(title="持股平均成本")
        self.investment.plot(title="投入资金")
        plt.show()

    def summary(self, write_excel=False):
        """
        Generate backtest report for the given strategy.
        """
        print("\n----- 策略回测报告 -----")

        profit_summary = pd.concat(
            [
                self.price,
                self.investment,
                self.share,
                self.average_cost,
                self.confirmed_profit,
                self.accumulated_profit,
                self.profit_today,
            ],
            axis=1,
        )
        profit_summary.fillna(method="ffill", inplace=True)
        if write_excel:
            profit_summary.to_excel("Position History.xlsx")

        print("累计投入\t{:.2f}".format(self.accumulated_investment.iloc[-1].sum())) # investment.T.sum().max()
        print("当前持仓总市值\t{:.2f}".format(self.capital_balance.iloc[-1].sum()+self.cash_balance))#confirmed_profit.iloc[-1,0]
        print("投入资金平均值\t￥{:.2f}".format(self.investment.T.sum().mean()))
        rate_of_return = self.accumulated_profit.iloc[-1,0]/self.investment.T.sum().mean()
        print("总收益率\t{:.2%}".format(rate_of_return))
        print("止盈参数\t{:.2%}".format(self.Profit_Ceiling[0]))

        return profit_summary

    @classmethod
    def generate_profit_curve(self, ans: pd.DataFrame, column="accumulated_profit"):
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
            ans.index, ans["NAV"], y2=1,
            where=(ans["NAV"] < ans["NAV"].shift(1))
            | ((ans["NAV"] > ans["NAV"].shift(-1)) & (ans["NAV"] >= ans["NAV"].shift(1))
            ),
            facecolor="grey", alpha=0.3,
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
            width, label="当日盈亏+",
            color="red", alpha=0.8,
        )
        bx.bar(
            ans.index,
            ans["Daily_pnl"].where(ans["Daily_pnl"] < 0),
            width, label="当日盈亏-",
            color="green", alpha=0.8,
        )
        bx.legend(fontsize=15)
        plt.grid()
        plt.show()



if __name__ == "__main__":
    pool = ["000016.SH","000905.SH","000009.SH","000991.SH","000935.SH","000036.SH"]
    # self.STOCK_POOL = ['000905.SH', '600309.SH', '600585.SH', '000538.SZ', '000651.SZ', '600104.SH', '600519.SH', '601888.SH']
        # self.STOCK_POOL = ["000036.SH"]
    test = Backtest(stock_pool=pool, type='index',
                    start_date="2008-01-01", end_date="2018-08-29", 
                    profit_ceiling=[0.5, 0.4, 0.2], trailing_percentage=[1, 0.3, 0.1])
                    # profit_ceiling=[0.5], trailing_percentage=[1])
    test.loop()
    Backtest.generate_profit_curve(test.summary())