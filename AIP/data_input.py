#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:21:52 2018

@author: mahaiqian
"""
#%%
import pymssql
import numpy
import pandas as pd
import sqlalchemy
import re
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
DBname = 'StrategyOutput'
engine = create_engine('mssql+pymssql://sa:abc123@192.168.0.28:1433/%s'%DBname,encoding = 'utf8')#数据库类型+驱动://用户名:密码@IP:端口/数据库名称

#DB_Session = sessionmaker(bind=engine)
#session = DB_Session()
def read_file(path):
    filetype = re.findall(r'.\.(.+\Z)',path)
    if filetype[0] == 'xlsx' or  filetype[0] == 'xls':
        df = pd.read_excel(path)
    elif filetype[0] == 'csv':
        df = pd.read_csv(path,engine = 'python')
    elif filetype[0] == ' txt':
        df = pd.read_table(path,',')
    return df

def view_table():
    sql = "select * from sysobjects  where xtype='U';"
    res = engine.execute(sql)
    table_list = pd.DataFrame(res.fetchall(),columns = res.keys())  
    return table_list

def input_SOdata(table_name,df,order='fail'):
    #order取值说明
    #'fail'：如果表格存在则报错
    #'replace'：替换已经存在表格
    #'append'：向已经存在的表格后新增数据
    tablelist = view_table()
    if table_name in list(tablelist.iloc[:,0]):
        if order == 'fail':
            return '表名重复'
        else:
            df.to_sql(table_name,engine,if_exists=order,chunksize=1000,index=False)
            return 'succeed!'
    else:
        df.to_sql(table_name,engine,if_exists=order,chunksize=1000,index=False)
        return 'succeed!'
 
def get_SOdata(table_name='mhqtable',columns = ['*'],time_range='no'):
    con=pymssql.connect(server='192.168.0.28',port=1433,user='sa',password='abc123',database=DBname,charset='utf8')
    #con = engine.connect()
    columns_name = ''.join([x+',' for x in columns])
    columns_name = columns_name[:-1]
    if time_range == 'no':
        sql = 'select %s from %s.dbo.%s'%(columns_name,DBname,table_name)
        data = pd.read_sql(sql,con)
        return data
    else:
        sqlt = "select column_name, data_type, LEN(data_type) from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME='%s'"%table_name
        data_time = pd.DataFrame(engine.execute(sqlt).fetchall())
        data_time.columns = ['name','type','length']
        try: 
            tline_name = data_time.iloc[data_time[(data_time.type=='datetime')].index.tolist()[0],0]
            sql = "select %s from %s.dbo.%s where %s between '%s' and '%s'"%(columns_name,DBname,table_name,tline_name,time_range[0],time_range[1])
            data = pd.read_sql(sql,con)   
            print('succeed!')
            con.close() 
            return data
        except IndexError as e:
            print(e,'没有时间列,请检查表格格式')


def delete_table(table_name = 'user',safe_order = 'on' ):
    #请非常慎用，为保险起见,safe_order='on'时将本地生成scv文件保存   
    if safe_order == 'on':
        safe_data = get_SOdata(table_name)
        safe_data.to_csv(table_name+'.csv',index=False)
    sql = 'drop table %s.dbo.%s'%(DBname,table_name)
    engine.execute(sql)
    return 'succeed!'

def del_rowORcolumns(table_name,limit,axis = 0 ):
    #用来删除某一行或某一列，输入表名，列名，删除行的对应值
    #axis = 0删除列，axis = 1，删除行
    #limit可变参数，删除列时为列名，删除行时为[列名，条件]，条件可以为字符串也可以为数字
    if axis == 1:
        if isinstance(limit[1],str):
            sql = "delete from %s.dbo.%s where %s = '%s';"%(DBname,table_name, limit[0], limit[1])
        else:
            sql = "delete from %s.dbo.%s where %s = %d;"%(DBname,limit[0], limit[1])
        
    else:
        sql = "alter table %s.dbo.%s drop column %s"%(DBname,table_name,limit)
        engine.execute(sql)
    return 'succeed!'

def view_tabletime(table_name,days='20'):
    #用于查看任意有时间列表格时间，days意味取多少最近时间
    con = engine.connect()
    sql = "select column_name, data_type, LEN(data_type) from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME='%s'"%table_name
    #sql2 = "select name as colName,type_name(xtype) as colType,length from syscolumns where id=object_id('%s')"%table_name
    data = pd.DataFrame(engine.execute(sql).fetchall())
    data.columns = ['name','type','length']
    ind = data[(data.type=='datetime')].index.tolist()[0]
    tline_name = data.iloc[ind,0]
    if days == 'all':
        sqlt = "select * from %s.dbo.%s Order By %s Desc"%(DBname,table_name,data[(data.type=='datetime')]['name'].tolist()[0])
        timetable = pd.read_sql(sqlt,con)
        timeline = timetable[tline_name]
    else:
        sqlt = "select top %d  * from %s.dbo.%s Order By %s Desc"%(DBname,days,table_name,data[(data.type=='datetime')]['name'].tolist()[0])
        timetable = pd.read_sql(sqlt,con)
        timeline = timetable[tline_name]
    return timeline

def add_columns(table_name, df):
    #用于在表格中添加列，可多列，要有时间列，放在第一列，时间序列要与数据库中一致
    time = df.sort_values(by = [df.iloc[:,0].name],axis = 0, ascending = False).iloc[:,0].reset_index(drop = True)
    dbtime = view_tabletime(table_name,'all')
    if time.size>dbtime.size:
        return '时间长度超出'
    elif time.size<dbtime.size:
        return '时间长度不足'
    elif ~(dbtime == time).all():
        return dbtime == time,'时间不匹配'
    else:
        input_SOdata('Add_Columns_Use_Table',df,order='replace')
        sql = "select column_name, data_type, LEN(data_type) from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME='Add_Columns_Use_Table'"
        typedf = pd.DataFrame(engine.execute(sql).fetchall())
        tableTcolumns = dbtime.name
        timecolumns = typedf.iloc[0,:].tolist()[0]
        typedf = typedf.drop(0).reset_index(drop = True)
        for rown in typedf.index:
            row = typedf.iloc[rown].tolist()
            sqladd = "alter table %s.dbo.%s add %s %s"%(DBname,table_name,row[0],row[1])
            engine.execute(sqladd)
            sqlupdate = "update a \
            set a.%s=b.%s \
            from [%s].[dbo].[%s] a,[%s].[dbo].[Add_Columns_Use_Table] b \
            where a.[%s] = b.[%s]"%(row[0],row[0],DBname,table_name,DBname,tableTcolumns,timecolumns)
            engine.execute(sqlupdate)
        delete_table('Add_Columns_Use_Table','off')
        return 'succeed!'
#def excute_sql(sql):
    #result = session.execute(sql)
    #return result
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    