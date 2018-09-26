# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:23:45 2018

@author: meiconte
"""

from Pyback1_2 import Backtest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#pool = ['600309.SH', '600585.SH', '000538.SZ', '000651.SZ', '600104.SH', '600519.SH', '601888.SH']
pool=['600000.SH','600016.SH','600019.SH','600028.SH','600029.SH','600030.SH','600036.SH','600048.SH','600050.SH','600104.SH','600111.SH','600309.SH','600340.SH','600518.SH','600519.SH','600547.SH','600606.SH','600837.SH','600887.SH','600919.SH','600958.SH','600999.SH','601006.SH','601088.SH','601166.SH','601169.SH','601186.SH','601211.SH','601229.SH','601288.SH','601318.SH','601328.SH','601336.SH','601390.SH','601398.SH','601601.SH','601628.SH','601668.SH','601669.SH','601688.SH','601766.SH','601800.SH','601818.SH','601857.SH','601878.SH','601881.SH','601985.SH','601988.SH','601989.SH','603993.SH']
# 主要借用了现成的test对象来获取数据，计算RSI
test = Backtest(pool=pool, type='stock', duration = 250,
                start_date="2008-02-01", end_date="2018-09-19", 
                load_data=True,
                profit_ceiling=[0.6,0.2], 
                trailing_percentage=[1, 0.2])

# share记录了实时的仓位信息
share = test.pos[test.pos.index.dayofweek == 4]
share = share.reindex(test.pos.index)
share = share.fillna(method='ffill')

price = test.price
price_pct_change = price.pct_change().replace(np.inf,0)
daily_pnl = price_pct_change * share
#分别绘制复利和单利
(daily_pnl.T.sum()+1).cumprod().plot()
(daily_pnl.T.sum().cumsum()+1).plot()
plt.show()