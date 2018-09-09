# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pymssql
import pandas as pd

conn=pymssql.connect(               #connect
	server='192.168.0.28',
	port=1433,
	user='sa',
	password='abc123',
	database='rawdata',
    charset='utf8'
	)

"""
input part example :

index='600000.SH'

start_date = '2016-01-01'

end_date = '2017-01-01'

freq = '3D' 

"""
def get_findata(SQL,write_to_file = 'off'):
    data = pd.read_sql(SQL,conn)
    if write_to_file == 'on':
        data.to_csv('data.csv')

    return data

def get_index_day(index,start_date,end_date,freq = '1D'):

    index_code2,index_code1 = index.split('.')

    SQL_code = "scode1="+"\'"+index_code1+"\' and scode2=\'"+index_code2+"\'"

    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"

    SQL_price = "select sclose,sopen,high,low,ddate from indexs_day where "+SQL_code+' and '+SQL_date

    SQL_vol = "select volumn,ddate from indexs_day where "+SQL_code+' and '+SQL_date

    price = pd.read_sql(SQL_price,conn,index_col='ddate')
    
    price = price.resample(freq).last().dropna()

    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()

    data = price.join(vol)
    
    return data

def get_index_min(index,start_date,end_date,freq = '1D'):

    index_code2,index_code1 = index.split('.')

    SQL_code = "scode1="+"\'"+index_code1+"\' and scode2=\'"+index_code2+"\'"

    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"

    SQL_price = "select sclose,sopen,high,low,ddate from indexs_5min where "+SQL_code+' and '+SQL_date

    SQL_vol = "select volumn,ddate from indexs_5min where "+SQL_code+' and '+SQL_date

    price = pd.read_sql(SQL_price,conn,index_col='ddate')
    
    price = price.resample(freq).last().dropna()

    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()

    data = price.join(vol)
    
    return data

def get_stock_min(index,start_date,end_date,freq = '5min'):

    index_code2,index_code1 = index.split('.')

    SQL_code = "scode1="+"\'"+index_code1+"\' and scode2=\'"+index_code2+"\'"

    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"

    SQL_price = "select sclose,sopen,high,low,ddate from stock_5min where "+SQL_code+' and '+SQL_date

    SQL_vol = "select volumn,ddate from stock_5min where "+SQL_code+' and '+SQL_date

    price = pd.read_sql(SQL_price,conn,index_col='ddate')
    
    if len(price.values) == 0:
        
        print('无相关数据，请检查该时间段内股票是否还未上市，或已退市')
        
        return -1
    
    price = price.resample(freq).last().dropna()

    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()

    data = price.join(vol)
    
    return data

def get_stock_day(index,start_date,end_date,freq = '1D'):

    index_code2,index_code1 = index.split('.')

    SQL_code = "scode1="+"\'"+index_code1+"\' and scode2=\'"+index_code2+"\'"

    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"

    SQL_price = "select sclose,sopen,high,low,ddate from stock_day where "+SQL_code+' and '+SQL_date

    SQL_vol = "select volumn,ddate from stock_day where "+SQL_code+' and '+SQL_date

    price = pd.read_sql(SQL_price,conn,index_col='ddate')
    
    if len(price.values) == 0:
        
        print('无相关数据，请检查该时间段内股票是否还未上市，或已退市')
        
        return -1
    
    price = price.resample(freq).last().dropna()

    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()

    data = price.join(vol)
    
    return data

def get_close_min(index,start_date,end_date,freq):
    
    index_code2,index_code1 = index.split('.')

    SQL_code = "scode1="+"\'"+index_code1+"\' and scode2=\'"+index_code2+"\'"

    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"
    
    SQL_price = "select sclose,ddate from stock_5min where "+SQL_code+' and '+SQL_date
    
    price = pd.read_sql(SQL_price,conn,index_col='ddate')
    
    if len(price.values) == 0:
        
        print(index+'无相关数据，请检查该时间段内股票是否还未上市，或已退市')
        
        return -1
    
    price = price.resample(freq).last().dropna()
    
    return price

def get_close_day(index,start_date,end_date,freq):
    
    # index_code2,index_code1 = index.split('.')
    start_date = start_date.strip('-')
    end_date = end_date.strip('-')
    SQL_code = "S_INFO_WINDCODE="+"\'"+index+"\'"# and scode2=\'"+index_code2+"\'"

    SQL_date = "TRADE_DT between "+start_date+" and " + end_date 
    
    SQL_price = "select TRADE_DT,S_DQ_ADJCLOSE from ASHAREEODPRICES where "+SQL_code+' and '+SQL_date
    
    price = pd.read_sql(SQL_price,conn)
    price.rename(columns={"TRADE_DT":'date','S_INFO_WINDCODE':'code',},inplace=True)
    price=price.astype({'date':'datetime64'})
    price.set_index('date',inplace=True)
    
    if len(price.values) == 0:
        
        print(index+'无相关数据，请检查该时间段内股票是否还未上市，或已退市')
        
        return -1
    
    price = price.resample(freq).last().dropna()
    
    return price
    
    
def get_muti_close_min(index,start_date,end_date,freq='5min'):
    
    muti_close = pd.DataFrame()
    
    for i in index:
        
        df = get_close_min(i,start_date,end_date,freq)
        
        if type(df) == int:
            
            continue
        
        df.columns = [i]
        
        muti_close = muti_close.join(df,how = 'outer')
        
    return muti_close

def get_muti_close_day(index,start_date,end_date,freq='1D'):
    
    muti_close = pd.DataFrame()
    
    for i in index:
        
        df = get_close_day(i,start_date,end_date,freq)
        
        if type(df) == int:
            
            continue
        
        
        df.columns = [i]
        
        muti_close = muti_close.join(df,how = 'outer')
        
    return muti_close

def gbfun(x):
    
    x = x.sort_values(by='volumn',ascending = False)
    
    return x.iloc[0]


def get_index_future_day(index_future,start_date,end_date,freq):
    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"
    
    SQL_future_index = "scode2 like \'"+index_future+"%\'"
    
    SQL =  "select ddate,sclose,sopen,high,low,volumn from indexfuture_day where "+SQL_date+" and "+SQL_future_index
    
    raw_data = pd.read_sql(SQL,conn,index_col='ddate')
    
    group = raw_data.groupby(by=raw_data.index).apply(gbfun)
    
    price = group[['sclose','sopen','high','low']]
    
    price = price.resample(freq).last().dropna()
    
    vol = group[['volumn']]
    
    vol = vol.resample(freq).sum().dropna()
    
    data = price.join(vol)
    
    return data


def get_index_future_min(index_future,start_date,end_date,freq):
    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"
    
    SQL_future_index = "scode2 like \'"+index_future+"%\'"
    
    SQL =  "select ddate,sclose,sopen,high,low,volumn from indexfuture_5min where "+SQL_date+" and "+SQL_future_index
    
    raw_data = pd.read_sql(SQL,conn,index_col='ddate')
    
    group = raw_data.groupby(by=raw_data.index).apply(gbfun)
    
    price = group[['sclose','sopen','high','low']]
    
    price = price.resample(freq).last().dropna()
    
    vol = group[['volumn']]
    
    vol = vol.resample(freq).sum().dropna()
    
    data = price.join(vol)
    
    return data

def get_comm_future_day(comm_future,start_date,end_date,freq):
    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"
    
    SQL_future_index = "scode2 like \'"+comm_future+"[0-9][0-9][0-9][0-9]\'"
    
    SQL =  "select ddate,sclose,sopen,high,low,volumn from commfuture_day where "+SQL_date+" and "+SQL_future_index
    
    raw_data = pd.read_sql(SQL,conn,index_col='ddate')
    
    group = raw_data.groupby(by=raw_data.index).apply(gbfun)
    
    price = group[['sclose','sopen','high','low']]
    
    price = price.resample(freq).last().dropna()
    
    vol = group[['volumn']]
    
    vol = vol.resample(freq).sum().dropna()
    
    data = price.join(vol)
    
    return data

def get_comm_future_min(comm_future,start_date,end_date,freq):
    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"
    
    SQL_future_index = "scode2 like \'"+comm_future+"[0-9][0-9][0-9][0-9]\'"
    
    SQL =  "select ddate,sclose,sopen,high,low,volumn from commfuture_5min where "+SQL_date+" and "+SQL_future_index
    
    raw_data = pd.read_sql(SQL,conn,index_col='ddate')
    
    group = raw_data.groupby(by=raw_data.index).apply(gbfun)
    
    price = group[['sclose','sopen','high','low']]
    
    price = price.resample(freq).last().dropna()
    
    vol = group[['volumn']]
    
    vol = vol.resample(freq).sum().dropna()
    
    data = price.join(vol)
    
    return data


def get_option_strike(expiry_date):
    
    year,month=expiry_date.split('-')
    
    SQL_expiry = "datepart(yy,expirydate)="+year+' and datepart(mm,expirydate)='+month
                          
    SQL = 'select * from option_day where '+SQL_expiry
    
    raw_data = pd.read_sql(SQL,conn,index_col='ddate')
    
    strike = raw_data.strike.unique()
    
    return strike


def get_option_day(expiry_date,strike,freq,type='C'):
    
    year,month=expiry_date.split('-')
    
    SQL_expiry = "datepart(yy,expirydate)="+year+' and datepart(mm,expirydate)='+month
                          
    SQL_strike = 'strike=' + str(strike)
    
    SQL_type = "tradecode like \'%"+type+"%\'"
                
    SQL = 'select * from option_day where '+SQL_expiry+' and '+SQL_strike+' and '+SQL_type
    
    raw_data = pd.read_sql(SQL,conn,index_col='ddate')
    
    price = raw_data[['strike','sclose','sopen','high','low','tradecode','contractunit','openinterest']]
    
    price = price.resample(freq).last().dropna()
    
    vol = raw_data[['volumn']]
    
    vol = vol.resample(freq).sum().dropna()
    
    data = price.join(vol)
    
    return data

def get_option_min(expiry_date,strike,freq,type='C'):
    
    year,month=expiry_date.split('-')
    
    SQL_expiry = "datepart(yy,expirydate)="+year+' and datepart(mm,expirydate)='+month
                          
    SQL_strike = 'strike=' + str(strike)
    
    SQL_type = "tradecode like \'%"+type+"%\'"
                
    SQL = 'select * from option_5min where '+SQL_expiry+' and '+SQL_strike+' and '+SQL_type
    
    raw_data = pd.read_sql(SQL,conn,index_col='ddate')
    
    price = raw_data[['strike','sclose','sopen','high','low','contractunit','openinterest']]
    
    price = price.resample(freq).last().dropna()
    
    vol = raw_data[['volumn']]
    
    vol = vol.resample(freq).sum().dropna()
    
    data = price.join(vol)
    
    return data
get_muti_close_day(['600309.SH'],'20180101','20180202','1D')
