# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:45:12 2018

@author: Administrator
"""

import datetime
from data_reader import get_index_min
index = "000016.SH"
start_date = "2000-01-01"
end_date = datetime.datetime.now().strftime('%Y-%m-%d') 
freq = "15min"
t = get_index_min(index, start_date, end_date, freq)

#%%
t.to_excel(r"C:/Users/Administrator/Desktop/data_reader/sz50.xlsx")