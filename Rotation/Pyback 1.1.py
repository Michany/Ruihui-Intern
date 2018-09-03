__doc__ = """
Pyback - an experimental module to perform backtests for Python
===============================================================

Main Features
-------------
Here are something this backtest framework can do:

- 选ROE始终>15且比较稳定，ROE位居行业内首位


Updated in 2018/9/3
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from collections import Iterable
from data_reader import get_muti_close_day, get_index_day
try:
    import talib
except ImportError as e:
    print(e)


class Backtest():
    """
    Backtest : Core class for a backtest.  
    -------------------------  

    ``Backtest(pool, type = 'stock',
             start_date="2008-01-01", end_date="2018-08-24", 
             duration=250, initial_capital=1000,
             load_data=True,
             profit_ceiling = [0.4,0.2], trailing_percentage=[1,0.2])``
    
    Parameters
    ----------
    pool: Your pool that you want to test

    type: 0 for "index"; 1 for "stock"

    load_data : If you want to use your own data, you can set ``load_data=False``, 
                and then set ``test.price = Your own Price DataFrame``

    Usage
    -----
    >>> test1 = Backtest(pool,
                start_date="2008-01-01", end_date="2018-08-24")
    >>> test1.StopLoss = 1
    >>> test.run()
    >>> test.summary
    """

    def __init__(self, pool, 
                 start_date="2008-01-01", end_date="2018-08-24", **arg):
        '''
        Example
        -------
        >>> Backtest(pool=["000016.SH","000905.SH","000009.SH","000991.SH","000935.SH","000036.SH"], 
            END_DATE='2018-8-31',type='index',duration = 250, 
            Profit_Ceiling = [0.4, 0.2], Trailing_Percentage=[1,0.2])
        '''
        self.START_DATE = start_date
        self.END_DATE = end_date
        self.POOL = pool

        initial_capital = arg.pop("initial_capital", None)
        if initial_capital is None:
            self.INITIAL_CAPITAL = 1000
        elif initial_capital <= 0:
            raise ValueError("Initial capital must be greater than zero.")
        else:
            self.INITIAL_CAPITAL = initial_capital

        # 设置止损止盈
        self.DURATION = arg.pop("duration", 250) # 设置duration必须放在设置price之前
        self.Profit_Ceiling = arg.pop("profit_ceiling", [0.4, 0.2])  # 止盈线
        self.Trailing_Percentage = arg.pop("trailing_percentage", [1, 0.2])    # 优先止盈百分比
        self.StopLoss = arg.pop("stoploss", 1)

        # 加载数据
        TYPE = arg.pop("type", None)
        if TYPE == 1 or TYPE == "stock":
            self.TYPE = 1
        elif TYPE == 0 or TYPE == "index":
            self.TYPE = 0
        else:
            raise ValueError("Type not recognized.")

        if arg.pop('load_data', True):
            self.init_data(type=self.TYPE)

    def init_data(self, type):
        """
        Fetch Data
        ----------
        获取收盘数据
        
        Type 0 for "index"; 1 for "stock"

        """
        print("Fetching Historical Data...")
        if type == 1:
            price = get_muti_close_day(self.POOL, self.START_DATE, self.END_DATE)
        elif type == 0:
            price = {}
            for symbol in self.POOL:
                price[symbol] = get_index_day(symbol, self.START_DATE, self.END_DATE).sclose
            price = pd.DataFrame(price)
        price.fillna(method="ffill", inplace=True)
        print("Historical Data Loaded!")
        
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
        '''
        The defalut value of the parameter "load_data" is True, 
        which means the historical price data will be loaded automatically from database.

        However, if you wish to use your own data to perform the backtest,
        you can simply set Backtest.price = _Your Price DataFrame_

        Note
        ----
        _Your Price DataFrame_ must be a instance of pd.DataFrame,
        and it must contain the columns of your stock pool.  
        Also, the index of _Your Price DataFrame_ should be <datetime64>.
        '''
        return self._price 
    @price.setter
    def price(self, price_DataFrame):
        if isinstance(price_DataFrame, pd.DataFrame):
            self._price = price_DataFrame
            self.refresh_TI() #给price赋值时，重新计算技术指标
            self._pos = self.timing()
        else:
            raise ValueError("Unable to handle the inputted price data of type {}, \
                             must be <pd.DataFrame>.".format(type(price_DataFrame)))
    @property
    def pos(self):
        return self._pos
    @pos.setter
    def pos(self, pos_DataFrame):
        if isinstance(pos_DataFrame, pd.DataFrame):
            self._pos = pos_DataFrame
        else:
            raise ValueError("Unable to handle the inputted position data of type {}, must be <pd.DataFrame>.".format(type(pos_DataFrame)))            
    
    @property
    def info(self):
        '''
        If you want to see the parameters of your current backtest, 
        you can simply run
        >>> test = Backtest()
        >>> test.info
        And the major parameters will be print on the screen.
        '''
        print('-'*5, 'Backtest Info', '-'*5)
        print(("Pool:\n"+'{} '*len(self.POOL)).format(*self.POOL))
        print("Duration\t{}".format(self.DURATION))
        print(("Profit_Ceiling\t"+"{:.0%}\t"*len(self.Profit_Ceiling)).format(*self.Profit_Ceiling))
        print(("Trailing_Pct\t"+"{:.0%}\t"*len(self.Trailing_Percentage)).format(*self.Trailing_Percentage))


    # %% 技术指标
    def refresh_TI(self, TI_list=[], 
                   TI_from_TA_lib={}):
        '''
        The technical indicators for your strategy will be calculated in this part.
        
        You can also use TA_lib directly.
        To do so, you have to input a certain dictionary, in the format of
        {"TI_name":(argument1, argument2, ...)}
        '''
        self._price.fillna(value=0, inplace=True)  # 把早年暂未上市的股票价格填充为0
        self._price_diff = self._price.diff()
        self._price_pct_chg = self._price.pct_change()
        self._price_std = self._price.rolling(window=self.DURATION).std()  # 计算标准差
        self._R = (self._price / self._price.shift(1)).apply(np.log)
        self._sigma = self._R.rolling(window=self.DURATION).std()
        self._mu = (self._price / self._price.shift(self.DURATION)).apply(np.log) + 0.5 * np.sqrt(self._sigma)
        
        for TI in TI_list:
            setattr(self, TI, 1)

        for TI in TI_from_TA_lib:
            try:
                if isinstance(TI_from_TA_lib[TI], Iterable):
                    temp = self.price.apply(getattr(talib, TI), args=TI_from_TA_lib[TI])
                else:
                    temp = self.price.apply(getattr(talib, TI), args=(TI_from_TA_lib[TI],))
                setattr(self, TI, temp)
            except Exception as e:
                if isinstance(e, AttributeError):
                    raise ValueError("Required technical indicator \'{}\' is not available in TA-lib.")
                elif isinstance(e, ValueError):
                    raise ValueError("Your parameters is not acceptable for TA-lib, please check once again.")
                else:
                    print(e)
                    print("Technical indicator \'{}\' will not be calculated.")
                    return


    def timing(self):
        # %% 策略部分 分配仓位
        # 买入时机
        is_entry_time = np.square(self._sigma * 1) - self._mu > -0.3
        percent_chg = (self.price / self.price.shift(int(self.DURATION / 2))) - 1

        # 赋权重(仓位)
        pos = self.price.copy()
        for index in self.POOL:
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

    def run(self, mute=False):
        """
        Loop is the main method for <Class Backtest>.
        ============================================

        变量命名及含义
        -----------
        - capital_balance           账户余额  
        - share                     持股份数
        - confirmed_profit          确认收益
        - accumulated_profit        累计收益（包含确认收益和浮动盈亏）
        - profit_today              当日盈亏
        - balance_today             当日账户余额（为账户余额的某一行）
        - share_today               当日持股份数（为持股份数的某一行）
        - win_count                 
        - lose_count  
        - average_cost              平均持仓成本
        - cash_balance              现金余额
        - 投入                       投入在某个标的上的资金
        - 实际投入                   实际投入的资金（优先用收回的资金再投入）
        - 累计投入                   累计投入在某个标的上的资金（递增）
        - investment                实际投入的DataFrame
        - accumulated_investment    累计投入的DataFrame


        """
        def init_run():
            # 添加首行（全为0）
            first = {}
            self.平均成本, self.累计投入, self.投入, self.实际投入 = {}, {}, {}, {}
            for symbol in self.POOL:
                first[symbol] = 0
                self.平均成本[symbol] = 10000
                self.累计投入[symbol] = 0
                self.投入[symbol] = 0
                self.实际投入[symbol] = 0
            self.capital_balance, self.share = [first], [first]

            self.confirmed_profit, self.profit_today = {}, {}  # 卖出后 已确认收益；继续持仓 浮动盈亏
            self.average_cost, self.investment, self.real_investment, self.accumulated_investment = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            self.win_count, self.lose_count = 0, 0

            self.ProfitTrailing_Already = pd.DataFrame(index=self.Profit_Ceiling,columns=self.POOL)
            self.ProfitTrailing_Already.fillna(False, inplace=True)

            self.cash_balance_today = 0
            self.cash_balance = pd.DataFrame(columns=['cash_balance'])

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
            else:
                amount = self.share[-1][symbol] * percentage
            if mute: end='\r'
            else: end='\n'
            print(day.strftime("%Y-%m-%d"), "卖出 {:.2f} 份 {}".format(amount, symbol),
                "({:.1%} 持仓)".format(percentage), "卖价 {:.2f}".format(price), end=end)

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
            self.实际投入[symbol] *= 1 - percentage
            self.cash_balance_today += amount * price
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
            
            if self.cash_balance_today>=amount * price:
                self.cash_balance_today -= amount * price
            else:
                self.实际投入[symbol] += amount * price-self.cash_balance_today
                self.cash_balance_today = 0

            print(
                day.strftime("%Y-%m-%d"),
                "买入 {:.1f}份 {}, 当前平均成本 {:.2f}".format(amount, symbol, self.平均成本[symbol]),
                end="         \r",
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
            for symbol in self.POOL:
                SELL_SIGNALS[symbol], BUY_SIGNALS[symbol] = {}, {}

                #止盈
                for i in range(len(self.Profit_Ceiling)):
                    if (price_today[symbol] > self.平均成本[symbol] * (1 + self.Profit_Ceiling[i]) 
                        and self.share[-1][symbol] > 0
                        and not self.ProfitTrailing_Already.loc[self.Profit_Ceiling[i],symbol]
                        ):
                        SELL_SIGNALS[symbol]["{:.0%}止盈".format(self.Profit_Ceiling[i])] = {
                            "symbol": symbol,
                            "percentage": self.Trailing_Percentage[i],
                            "price": price_today[symbol],
                        }
                        # 记录该种止盈条件已经止盈过了
                        self.ProfitTrailing_Already.loc[self.Profit_Ceiling[i],symbol] = True
                        # 如果是全部止盈(i=0)，则将之前记录清空
                        if i==0:
                            self.ProfitTrailing_Already.loc[:,symbol] = False
                            # 重新审查
                        # break语句 保证每次只有一种止盈
                        break
                #止损

                #买入（仅在没有卖出信号下买入）
                if len(SELL_SIGNALS[symbol]) == 0:
                    if self.pos.loc[day, symbol] > 0:
                        BUY_SIGNALS[symbol]["定投买入"] = {
                            "symbol": symbol,
                            "amount": self.pos.loc[day, symbol],
                            "price": price_today[symbol],
                        }
                        total_share_required += self.pos.loc[day, symbol]
            # 将买入的总金额调整至每期固定投入 INITIAL_CAPITAL
            for symbol in self.POOL:
                try:
                    BUY_SIGNALS[symbol]["定投买入"]["amount"] *= (
                        (self.INITIAL_CAPITAL)/ total_share_required / price_today[symbol])
                except:
                    pass
            return BUY_SIGNALS, SELL_SIGNALS

        # 按天回测正式开始
        init_run()
        t0 = time.time()
        for day in self.price.index[self.DURATION:]:
            self.balance_today, self.share_today = {"date": day}, {"date": day}
            buy_signals, sell_signals = Check_Signal(self.price.loc[day])
            # print(buy_signals, sell_signals)
            for symbol in self.POOL:
                for signal in sell_signals[symbol]:
                    Sell(**sell_signals[symbol][signal])
                for signal in buy_signals[symbol]:
                    Buy(**buy_signals[symbol][signal])
                if (len(buy_signals.get(symbol, {})) == 0 and len(sell_signals.get(symbol, {})) == 0):
                    record_daily_profit(day, symbol)
                self.average_cost.loc[day, symbol] = (self.平均成本[symbol] if self.平均成本[symbol] < 10000 else np.nan)
                self.investment.loc[day, symbol] = self.投入[symbol]
                self.real_investment.loc[day, symbol] = self.实际投入[symbol]
                self.accumulated_investment.loc[day, symbol] = self.累计投入[symbol]
            # 记录当日份数和账户余额
            self.share.append(self.share_today)
            self.capital_balance.append(self.balance_today)
            self.cash_balance.loc[day] = self.cash_balance_today
        del self.share_today, self.balance_today
        t1 = time.time()
        tpy = t1 - t0
        print("\n回测已完成，用时 %5.3f 秒" % tpy)

        
        self.share = pd.DataFrame(self.share).set_index("date")
        self.capital_balance = pd.DataFrame(self.capital_balance).set_index("date")

        def dict_to_df(d: dict, columns_name: str):
            return pd.DataFrame(list(d.values()), index=d.keys(), columns=[columns_name])

        self.profit_today = dict_to_df(self.profit_today, "profit_today")
        self.accumulated_profit = self.profit_today.cumsum().rename(columns={"profit_today": "accumulated_profit"})
        # confirmed_profit 原先是每天的确认收益，现在用.cumsum() 转成累计的
        self.confirmed_profit = dict_to_df(self.confirmed_profit, "confirmed_profit").cumsum()

        self.average_cost.plot(title="持股平均成本")
        self.real_investment.plot(title="投入资金")
        plt.show()


    def summarize(self, write_excel=False):
        """
        Summary procedure for a looped backtest 
        --------------------------------------

        - Generate backtest report for the given strategy.
        - Return a DataFrame that describe and record the history of the backtest.
        """
        # 防止没有loop就总结
        try:
            type(self.accumulated_profit)
        except:
            raise RuntimeError("")

        print()
        print("====== 策略回测报告 ======")
        print("------  投资时间  ------")
        print("{} ~ {}".format(self.START_DATE,self.END_DATE))


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
        print("------  业绩表现  ------")
        print("实际投入资金\t{:.2f}".format(self.real_investment.T.sum().max())) # accumulated_investment.iloc[-1].sum()
        print("当前持仓总市值\t{:.2f}".format(self.capital_balance.iloc[-1].sum()+self.cash_balance_today))#confirmed_profit.iloc[-1,0]
        # print("投入资金平均值\t￥{:.2f}".format(self.investment.T.sum().mean()))
        # rate_of_return = self.accumulated_profit.iloc[-1,0]/self.investment.T.sum().mean()
        self.rate_of_return = (self.capital_balance.iloc[-1].sum()+self.cash_balance_today)/self.real_investment.T.sum().max()-1
        print("总收益率\t{:.2%}".format(self.rate_of_return))
        print("资金利用效率")

        print("------  回撤情况  ------")
        self.drawdown()
        print("最大回撤 {:.2f}, {} ~ {}".format(*self.max_draw_down))
        print("----  逐年收益及回撤  ----")
        print("年份\t收益\t\t回撤\t起止时间")
        for year in self.yearly_drawdown:
            print("{}\t{:.2f}\t\t{:.2%}\t{} ~ {}".format(year, self.yearly_return[year], *self.yearly_drawdown[year]))
        print("------  参数设定  ------")
        self.info

        print("="*25)
        return profit_summary

    @property
    def summary(self):
        self.summarize()

    def drawdown(self):
        '''
        计算内容：最大回撤比例，累计收益率  
        计算方式：单利  
        返回结果：最大回撤率，开始日期，结束日期，总收益率，年化收益，年化回撤  
        '''
        warnings.simplefilter("ignore")
        t0=time.time()

        t =self.accumulated_profit.rename(columns={'accumulated_profit':'Capital'})
        t['date'] = self.accumulated_profit.index
        t['Year']=t['date'].apply(lambda x:x.year)
        t.set_index('date', inplace=True)

        # 按年统计
        yearly_drawdown = dict()
        yearly_return = dict() # 记录年收益
        t_group=t.groupby('Year')
        year_groups=[t_group.get_group(i) for i in t_group.groups.keys()]
        for year_group in year_groups:
            max_draw_down, temp_max_value = 0, year_group['Capital'][0]
            start_date, end_date, current_start_date = year_group.index[0], year_group.index[-1], year_group.index[0]
            continous = False # 是否连续
            for day in year_group.index:
                if temp_max_value < year_group.loc[day,'Capital']:
                    current_start_date = day
                    temp_max_value = max(temp_max_value, year_group.loc[day,'Capital'])
                    continous = False
                else:
                    if max_draw_down>year_group.loc[day,'Capital']/temp_max_value-1:
                        if not continous: 
                            continous = True
                        max_draw_down = year_group.loc[day,'Capital']/temp_max_value-1
                    else:
                        if continous:
                            continous = False
                            start_date = current_start_date
                            end_date = day
            yearly_drawdown[day.year] = max_draw_down, start_date, end_date
            yearly_return[day.year] = year_group['Capital'][-1] - year_group['Capital'][0]

        #统计整体
        max_draw_down, temp_max_value = 0, t['Capital'][0]
        start_date, end_date, current_start_date = t.index[0], t.index[-1], t.index[0]
        continous = False # 是否连续
        for day in t.index:
            if temp_max_value < t.loc[day,'Capital']:
                current_start_date = day
                temp_max_value = max(temp_max_value, t.loc[day,'Capital'])
                continous = False            
            if max_draw_down>t.loc[day,'Capital']-temp_max_value:#t.loc[day,'Capital']/temp_max_value-1:
                if not continous: 
                    continous = True
                max_draw_down = t.loc[day,'Capital']-temp_max_value#t.loc[day,'Capital']/temp_max_value-1
            else:
                if continous:
                    continous = False
                    start_date = current_start_date
                    end_date = day
        t1 = time.time()
        tpy = t1 - t0
        print("回撤计算已完成，用时 %5.3f 秒" % tpy)
        self.max_draw_down = (max_draw_down, start_date, end_date)#.strftime("%Y/%m/%d")
        self.yearly_drawdown = yearly_drawdown
        self.yearly_return = yearly_return

    @classmethod
    def generate_profit_curve(self, data: pd.DataFrame, 
                            columns=["accumulated_profit","confirmed_profit"]):
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        print("正在生成图像...")
        fig = plt.figure()
        fig.set_size_inches(12, 12)
        ans = pd.DataFrame(index=data.index)
        ans["NAV"] = data[columns[0]]
        ans["Daily_pnl"] = data[columns[0]].diff()
        ans["confirmed_profit"] = data[columns[1]]

        ax = fig.add_subplot(211)
        ax.plot(ans.index, ans["NAV"], linewidth=2, label="累计收益")
        ax.plot(ans.index, ans["confirmed_profit"], linewidth=2, label="确认收益")
        # 最大回撤标注
        ax.fill_between(
            ans.index, ans["NAV"], y2=1,
            where=(ans["NAV"] < ans["NAV"].shift(1))
            | ((ans["NAV"] > ans["NAV"].shift(-1)) & (ans["NAV"] >= ans["NAV"].shift(1))),
            facecolor="grey", alpha=0.3,)
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

# TODO Parameter Optimizer
class Optimizer():
    def __init__(self, **arg):
        pass
    @classmethod
    def pool_optimize(self, backtest, pool):
        backtest.info
        for symbol in pool:
            single_pool = list()
            single_pool.append(symbol)
            backtest.POOL = single_pool
            backtest.run(mute=True)
            backtest.summary
        return
    def step_in_optimize(self):
        return
    def auto_optimize(self):
        return
    def random_optimize(self):
        return
    def set_parameters(self):
        return
    


if __name__ == "__main__":
    pool = ["000016.SH","000905.SH","000009.SH","000991.SH","000935.SH","000036.SH"]
    # pool = ['600309.SH', '600585.SH', '000538.SZ', '000651.SZ', '600104.SH', '600519.SH', '601888.SH']
    # pool = ["000036.SH"]
    test = Backtest(pool=pool, type='stock', duration = 250,
                    start_date="2007-02-01", end_date="2018-08-29", 
                    load_data=False,
                    profit_ceiling=[0.6, 0.5, 0.4, 0.3, 0.2], 
                    trailing_percentage=[1,0.8, 0.6, 0.4, 0.2])
                    # profit_ceiling=[0.5], trailing_percentage=[1])
    t = pd.read_excel(r"C:\Users\\meiconte\Documents\RH\Historical Data\指数价2007.xlsx",
                      dtype={'Date': 'datetime64'})
    price = t.set_index("Date")
    test.price = price
    test.run(mute=True)
    Backtest.generate_profit_curve(test.summarize())
    Optimizer.pool_optimize(backtest=test,pool=test.POOL)