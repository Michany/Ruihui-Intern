# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:53:53 2018

@author: Administrator
"""

import pymssql
import pandas as pd

conn=pymssql.connect(               #connect
	server='192.168.0.28',
	port=1433,
	user='sa',
	password='abc123',
    database='StrategyOutput'
	)

cursor = conn.cursor()

sql = "SELECT NAME FROM SYSOBJECTS WHERE TYPE='U'"

cursor.execute(sql)

x = cursor.fetchall()

x.sort()