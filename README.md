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
  
2. Similarity 
  __An experimental program to find similar patterns in history price series__  
  (Mostly wrote in Berkely, US in July 2018)  
   _stable version: similarity_remote_data.py_   
   
  <p align="center">
  <img src="https://github.com/Michany/Ruihui-Intern/raw/master/Similarity/curve.png" alt="Sample"  width="600" height="450">
  </p>
  
3. AIP  
   __A foundation research about Auto Investment Plan (AIP)__  
   It is a good time to try AIP (定投), espeacially when Chinese stock market was in a bear market.  
   (Partially overlap with _Rotation_)
  
4. Rotation   
   __A mixed strategy: apply AIP method on sector rotation strategy__  
    - stock/index Pool: _simple version_  
      Rotate in a given pool. This could cause some bias.  
    - ROE selection:  _complex version_  
      Use ROE criterias to select stocks in each sectors first, then apply AIP method.  
    - Round the shares by 2 or even less to make it feasible   
    <p align="center">
    <img src="https://github.com/Michany/Ruihui-Intern/raw/master/Rotation/定投保留1位小数10-10.png" alt="Sample"  width="600" height="500">
    </p>
    
   _stable version: Pyback roe选股.py_  
  
5. Options (Currently working on)  
   Designing a product for those who already hold the long position of ZE901 (core)  
   The decision of buying or writing an option is made based on the prediction of the core price.  
   
6. Index enhancement (Currently working on)  
   __A allocation strategy: based on RSI cross section__  
   
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
    The enhancement is made mainly based on the RSI cross section, with some sort of timing decision. I assume the trasaction fee is 1.3‰ double sides.  
    It turns out RSI can be a very effective technical indicator, if used on a pool with neither too many stocks nor too less.  
    
    Some highlighted features:
    - __Could be rounded to 100 shares.__ That means, with only ￥1 million, you can actually apply the strategy.
    - __Low ratio of turnover.__ No need to adjust position in high frequency. Actually you need to do it only weekly.  
    - __Thursday is a good day.__  
    
    _stable version: summary6.2_bitCoin.py_
    
7. RNN / LSTM (Currently working on)
   Pytorch


## Some thoughts
1. Which price should I use in a backtest?
   > For the following three prices:
     - When performing backtest, use forward-adjusted (前复权) price.  
     - Do not use backward-adjusted (后复权) price.
     - Use non-adjusted price when performing backtest for T0 trading strategy.
   
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
