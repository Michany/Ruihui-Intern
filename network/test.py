# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:45:17 2018

@author: Administrator
"""
from datetime import datetime
import pymssql
import pandas as pd

conn=pymssql.connect(               #connect
	server='192.168.0.28',
	port=1433,
	user='sa',
	password='abc123',
	database='rawdata'
	)

cursor = conn.cursor()

table = input('请输入关联表\n\n')

opinion = input('请输入观点\n\n')

date = str(datetime.today())[:19]
print(date)

putstring="(\'"+date+"\',\' "+opinion+"\', \'"+table+"\')"

print(putstring+'\n')

c = input('是否存入数据库 y/n \n')



sql = 'INSERT INTO 描述观点 VALUES '+putstring

cursor.execute(sql)

if c == 'y':
    conn.commit()
    