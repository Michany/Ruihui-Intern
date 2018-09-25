# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:09:48 2018

@author: Administrator
"""

import pymssql
import pandas as pd

conn=pymssql.connect(               #connect
	server='192.168.0.28',
	port=1433,
	user='sa',
	password='abc123',
   database='StrategyOutput',
   charset ='gbk'
	)

cursor = conn.cursor()

cursor.execute(sql)

x = cursor.fetchall()

