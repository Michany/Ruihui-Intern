# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:09:05 2018

@author: Administrator
"""

import sys
local_path = 'C://Users//Administrator/Desktop/'
if local_path not in sys.path:
    sys.path.append(local_path)
from get_data import data_reader as rd
import pymssql


conn=pymssql.connect(               #connect
	server='192.168.0.28',
	port=1433,
	user='sa',
	password='abc123',
	database='rawdata'
	)

cursor = conn.cursor()

sql = "select scode1,scode2 where ddate ='2017-01-05'"

cursor.execute(sql)

x = cursor.fetchall()