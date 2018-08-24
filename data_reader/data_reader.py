# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pymssql
import pandas as pd
import numpy as np

conn=pymssql.connect(               #connect
	server='192.168.0.28',
	port=1433,
	user='sa',
	password='abc123',
	database='rawdata'
	)

"""
input part example :

index='600000.SH'

start_date = '2016-01-01'

end_date = '2017-01-01'

freq = '3D' 

"""

def get_index_day(index,start_date,end_date,freq = '1D'):

    index_code2,index_code1 = index.split('.')

    SQL_code = "scode1="+"\'"+index_code1+"\' and scode2=\'"+index_code2+"\'"

    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"

    SQL_price = "select sclose,sopen,high,low,ddate from indexs_day where "+SQL_code+' and '+SQL_date

    SQL_vol = "select volumn,ddate from indexs_day where "+SQL_code+' and '+SQL_date

    price = pd.read_sql(SQL_price,conn,index_col='ddate')
    
    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()
    
    price = pd.concat([close,open,high,low],axis=1)

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

    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()
    
    price = pd.concat([close,open,high,low],axis=1)

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
    
    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()
    
    price = pd.concat([close,open,high,low],axis=1)

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
    
    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()
    
    price = pd.concat([close,open,high,low],axis=1)

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
    
    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    price = pd.concat([close,open,high,low],axis=1)

    
    return price

def get_close_day(index,start_date,end_date,freq):
    
    index_code2,index_code1 = index.split('.')

    SQL_code = "scode1="+"\'"+index_code1+"\' and scode2=\'"+index_code2+"\'"

    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"
    
    SQL_price = "select sclose,ddate from stock_day where "+SQL_code+' and '+SQL_date
    
    price = pd.read_sql(SQL_price,conn,index_col='ddate')
    
    if len(price.values) == 0:
        
        print(index+'无相关数据，请检查该时间段内股票是否还未上市，或已退市')
        
        return -1
    
    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    
    price = pd.concat([close,open,high,low],axis=1)
    
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
    
    SQL =  "select ddate,sclose,sopen,high,low,volumn,settle1 from indexfuture_day where "+SQL_date+" and "+SQL_future_index
    
    raw_data = pd.read_sql(SQL,conn,index_col='ddate')
    
    group = raw_data.groupby(by=raw_data.index).apply(gbfun)
    
    price = group[['sclose','sopen','high','low','settle1']]
    
    close = price.sclose.resample(freq).last().dropna()
    
    settle = price.settle1.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    price = pd.concat([close,open,high,low,settle],axis=1)
    
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
    
    close = price.sclose.resample(freq).last().dropna()
    
    settle = price.settle1.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    price = pd.concat([close,open,high,low,settle],axis=1)
    
    vol = group[['volumn']]
    
    vol = vol.resample(freq).sum().dropna()
    
    data = price.join(vol)
    
    return data

def get_comm_future_day(comm_future,start_date,end_date,freq):
    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"
    
    SQL_future_index = "scode2 like \'"+comm_future+"[0-9][0-9][0-9][0-9]\'"
    
    SQL =  "select ddate,sclose,sopen,high,low,volumn,settle1 from commfuture_day where "+SQL_date+" and "+SQL_future_index
    
    raw_data = pd.read_sql(SQL,conn,index_col='ddate')
    
    group = raw_data.groupby(by=raw_data.index).apply(gbfun)
    
    price = group[['sclose','sopen','high','low','settle1']]
    
    settle = price.settle1.resample(freq).last().dropna()
    
    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    price = pd.concat([close,open,high,low,settle],axis=1)
    
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
    
    close = price.sclose.resample(freq).last().dropna()

    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    price = pd.concat([close,open,high,low,settle],axis=1)
    
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

    strike = [i for i in strike if i*100 % 5 ==0]
    
    strike.sort()
   
    for i in strike:
        print(round(i,2))


def get_option_day(expiry_date,strike,freq,type='C'):
    
    year,month=expiry_date.split('-')
    
    SQL_expiry = "datepart(yy,expirydate)="+year+' and datepart(mm,expirydate)='+month
                          
    SQL_strike = 'strike=' + str(strike)
    
    SQL_type = "tradecode like \'%"+type+"%\'"
                
    SQL = 'select * from option_day where '+SQL_expiry+' and '+SQL_strike+' and '+SQL_type
    
    raw_data = pd.read_sql(SQL,conn,index_col='ddate')
    
    price = raw_data[['strike','sclose','sopen','high','low','tradecode','contractunit','openinterest','scode']]
    
    close = price.sclose.resample(freq).last().dropna()
    
    scode = price.scode.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    strike = price.strike.resample(freq).last().dropna()
    
    cu = price.contractunit.resample(freq).last().dropna()
    
    oi = price.openinterest.resample(freq).last().dropna()
    
    price = pd.concat([close,open,high,low,cu,oi,scode],axis=1)
    
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
    
    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    strike = price.strike.resample(freq).last().dropna()
    
    cu = price.contractunit.resample(freq).last().dropna()
    
    oi = price.openinterest.resample(freq).last().dropna()
    
    price = pd.concat([close,open,high,low,cu,oi],axis=1)
    
    vol = raw_data[['volumn']]
    
    vol = vol.resample(freq).sum().dropna()
    
    data = price.join(vol)
    
    return data

def get_HKindex_min(index,start_date,end_date,freq = '1D'):


    SQL_code = "scode="+"\'"+index+".HI\'"

    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"

    SQL_price = "select sclose,sopen,high,low,ddate from hsi_hscei_1min where "+SQL_code+' and '+SQL_date

    SQL_vol = "select volumn,ddate from hsi_hscei_1min where "+SQL_code+' and '+SQL_date

    price = pd.read_sql(SQL_price,conn,index_col='ddate')
    
    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()
    
    price = pd.concat([close,open,high,low],axis=1)

    data = price.join(vol)
    
    return data

def get_HKfuture_index_min(index,start_date,end_date,freq = '1D'):


    SQL_code = "scode="+"\'"+index+".HK\'"

    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"

    SQL_price = "select sclose,sopen,high,low,ddate from hhif_hsif_1min where "+SQL_code+' and '+SQL_date

    SQL_vol = "select volumn,ddate from hhif_hsif_1min where "+SQL_code+' and '+SQL_date

    price = pd.read_sql(SQL_price,conn,index_col='ddate')
    
    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()
    
    price = pd.concat([close,open,high,low],axis=1)

    data = price.join(vol)
    
    return data

def get_a50_min(start_date,end_date,freq = '1D'):


    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"

    SQL_price = "select sclose,sopen,high,low,ddate from a50_1min where "+SQL_date

    SQL_vol = "select volumn,ddate from a50_1min where "+SQL_date

    price = pd.read_sql(SQL_price,conn,index_col='ddate')
    
    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()
    
    price = pd.concat([close,open,high,low],axis=1)

    data = price.join(vol)
    
    return data

def get_50etf_day(start_date,end_date,freq='1D'):
    
    SQL_date ="date between "+"\'"+start_date+"\' and \'" + end_date+"\'"

    SQL_price = "select * from etfAllFull where "+SQL_date

    price = pd.read_sql(SQL_price,conn,index_col='date')
    
    t = pd.datetime.strptime
    
    t = np.vectorize(t)
    
    price.index = t(price.index,"%Y-%m-%d")
    
    price = price.resample(freq).last().dropna()
    
    return price