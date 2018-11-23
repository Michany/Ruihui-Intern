# Ruihui-Intern
_Intern: Michael Wang, From March 2018 so far_

## Projects during Ruihui Internship...  
Projects listed as follows (in chronical order):

1. Summary  
    __A pratical program to summarize a strategy visually__  

    Based on Tradestaion Excel output files, or other files that have at least two columns (Daily P&L, NAV) with a datetime index.  
    It includes servel modules that can be easily transfered into other programs.  

    - ``generate_profit_graph(NAV:pd.Series, Daily_pnl:pd.Series)``  
      Generate a profit curve with daily P&L under it.  
      You shall see a graph in the following pages.
    - ``drawdown_by_time(NAV:pd.Series)``  
      Calculate the maximum drawdown both groupby year and in total.  
      Also record the start and the end of the drawdown.  
    - ``write_to_excel()``
      Write a pretty-looking table into an Excel Worksheet. 

   _stable version: summary6.2_bitCoin.py_  

2. Hurst Exponent  
   __A technical indicator to calculate the stochastic `Cycle` in historical stock price series__

   Available in TradeStation EasyLanguage code.
  
   ref: https://xueqiu.com/8572240683/69748500

-------------------------------------------------------------- 
3. Similarity  
  __An experimental program to find similar patterns in history price series__  
  (Mostly wrote in Berkely, US in July 2018)  
  
   _stable version: similarity_remote_data.py_   
   
  <p align="center">
  <img src="https://github.com/Michany/Ruihui-Intern/raw/master/Similarity/curve.png" alt="Sample"  width="500" height="350">
  </p>
  
-------------------------------------------------------------- 
4. AIP  
   __A foundation research about Auto Investment Plan (AIP)__  
   It is a good time to try AIP (定投), espeacially when Chinese stock market was in a bear market.  

   We run it on a weekly basis, with ￥1,000 every round.

   _stable version: StockPool 轮动 1.5.py_  
   (Partially overlap with _Rotation_)
  
-------------------------------------------------------------- 
5. Rotation   
   __A mixed strategy: apply AIP method on sector rotation strategy__  
    - stock/index Pool: _simple version_  
      Rotate in a given pool. This could cause some bias.  
    - ROE selection:  _complex version_  
      Use ROE criterias to select stocks in each sectors first, then apply AIP method.  
    - Round the shares by 2 or even less to make it feasible   
    <p align="center">
    <img src="https://github.com/Michany/Ruihui-Intern/raw/master/Rotation/定投保留1位小数10-10.png" alt="Sample"  width="500" height="450">
    </p>
    
   _stable version: Pyback roe选股.py_  
  
-------------------------------------------------------------- 
6. Options (Currently working on)  
   Designing a product for those who already hold the long position of ZE901 (core)  

   The decision of buying or writing an option is made based on the prediction of the core price.  
  
-------------------------------------------------------------- 
7. Some VBA worksheets/programs for work effciency  
   - 水滴定投计划A.xlsx
   
-------------------------------------------------------------- 
8. Index enhancement (Currently working on)  
   __A allocation strategy: based on RSI cross section__  
   The enhancement is made mainly based on the RSI cross section, with some sort of timing decision.  
   I assume the trasaction fee is 1.3‰ double sides.  
   It turns out RSI can be a very effective technical indicator, if used on a pool with neither too many stocks nor too less.  

   Work out for the following indexes:  
    - HS300
    - ZZ500
    - HSI / stocks in Hong Kong with market value greater than ￥50 billion.
    - _TODO: apply it in US stock market_  
    
    <p align="center">
        <img src="https://github.com/Michany/Ruihui-Intern/raw/master/IndexEnhancement/HS300.png" alt="Sample"  width="500" height="330">
        <p align="center">
            <em>Performance</em>
        </p>
    </p>    


   Features
   --------
    - __Could be rounded to 100 shares.__ That is to say, with only ￥1 million, you can actually apply the strategy.
    - __Low ratio of turnover.__ No need to adjust position in high frequency. Actually you need to do it only weekly.  
    - __Thursday is a good day.__  
   
   Bug fixed
   ---------
    - 每个月初
    - pnl计算
    - TODO：成份股调整
   
   Limitations
   -----------
    - __Can hardly beat the soar market.__   
      If the stock market soars, this strategy can only keep the same profiting path at the best.  
      Although we can adjust parameters to address to this problem, it will harm the performance at the bear market.  
      Unless there's coresponding hedge strategy.

      __Maybe it is because $\alpha$ is missed__

    - __A lot of fees.__  
      They say sometimes fee is a good thing when negotiating with the dealers/brokers...
  
    Two Branches
    ------------
    - Low frequency: weekly, daily  
      _stable version: RSI横截面选股1.2 周频.py_  
      
    - Various bugs fixed:  
      _stable version: RSI横截面选股1.2 日频 每年重置 local.py_
  
    - High frequency: 1 hour  
      _stable version: RSI横截面选股min.py_
    

-------------------------------------------------------------- 
9. RNN / LSTM (Currently working on)  
   Pytorch

   Alpha should probably be the target.

-------------------------------------------------------------- 

## Some thoughts

1. Which price should I use in a backtest?
   > For the following three prices:
   > - When performing backtest, use forward-adjusted (前复权) price.  
   > - Do not use backward-adjusted (后复权) price.
   > - Use non-adjusted price when performing backtest for T0 trading strategy.
   
2. Differences and midunderstandings about "delta" and "gross".
   > Under most circumstances, "delta" measures the rick exposure of your position.  
   > However, it is not always the case when you have both long and short position (on slightly different underlying assets, which you thought are the same).
   
3. How to calculate __accumulated mean__ quickly?
   > Simple.  
   > Although there is no direct method like ``cummean()``, you can still make full use of ``rolling(window=10000, min_period=0).mean()``. That is a quick method using C++ based code.  
   
4. The use of Auto-Encoder.
   > An non-linear way to do PCA, and reduce dimensionality.

5. Careful when you use position to perform backtest.
    > A common practice is to multiply ``pos`` and ``price.pct_change()``, but the total position should have changed every row.  
    > To avoid inaccuracy, a feasible method is to transfer ``pos`` into ``share``.
    > Also, you can use ``pos.sum()\=pos.sum().shift(1)``
