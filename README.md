# Ruihui-Intern
_Intern: Michael Wang, From March 2018 so far_
## Projects during Ruihui Internship...  
Projects listed as follows (in chronical order):
1. Summary  
  __A pratical program to summarize a strategy visually__  
  Based on Tradestaion Excel output files, or other files that have at least two columns (Daily P&L, NAV) with a datetime index.  
  It includes servel modules that can be easily transfered into other programs.  
    - generate_profit_graph()
    - drawdown_by_time()
    - ...
  stable version: summary6.2_bitCoin.py  
2. Similarity 
  __An experimental program to find similar patterns in history price series__  
  (Mostly wrote in Berkely, US in July 2018)  
  stable version: similarity_remote_data.py  
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
  ![image](https://github.com/Michany/Ruihui-Intern/raw/master/Rotation/定投保留1位小数10-10.png)
5. Index enhancement  
    - HS300
    - ZZ500
  Based on RSI cross section  
  ![image](https://github.com/Michany/Ruihui-Intern/raw/master/IndexEnhancement/HS300.png)   
  __Also need to be rounded.__
6. RNN / LSTM (Currently working on)


## Some thoughts
1. Which price should I use?
   For the following three prices:
   - When performing backtest, use forward-adjusted (前复权) price.  
   - Do not use backward-adjusted (后复权) price.
   - Use non-adjusted price when performing backtest for T0 trading strategy.
2. Differences and midunderstandings about "delta" and "gross".
   Under most circumstances, "delta" measures the rick exposure of your position.  
   However, it is not always the case when you have both long and short position (on slightly different underlying assets, which you thought are the same).
3. The use of Auto-Encoder.
4. 
