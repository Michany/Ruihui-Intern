import pandas as pd
from WindPy import *
import time


def getCurrentTime():
    return time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))


def get_stock(stock_symbol, start, end):
    global index_data
    print('Initializing:')
    w.start()
    print(getCurrentTime(), stock_symbol, ": Download Starting...")
    try:
        stock = w.wsd(stock_symbol, 
                      "trade_code,open,high,low,close,pre_close,volume,amt,dealnum,chg,pct_chg,vwap, adjfactor,close2,turn,free_turn,oi,oi_chg,pre_settle,settle,chg_settlement,pct_chg_settlement, lastradeday_s,last_trade_day,rel_ipo_chg,rel_ipo_pct_chg,susp_reason,close3, pe_ttm,val_pe_deducted_ttm,pe_lyr,pb_lf,ps_ttm,ps_lyr,dividendyield2,ev,mkt_cap_ard,pb_mrq,pcf_ocf_ttm,pcf_ncf_ttm,pcf_ocflyr,pcf_nflyr,trade_status",
                      start, end)
        '''
        {'ev','总市值1';'pe_ttm','市盈率PE(TTM)';'pb_lf','市净率';'ps_ttm','市销率';
        'pcf_ocf_ttm','市现率';'dividendyield2','股息率近12个月';'mkt_freeshares','自由流通市值'};
        '''
        print(getCurrentTime(), stock_symbol, 'Download sucessful!')
    except:
        print(getCurrentTime(), stock_symbol, ": Download Error!")
    try:
        index_data = pd.DataFrame()
        index_data['trade_date'] = stock.Times
        stock.Data[0] = stock_symbol
        index_data['stock_code'] = stock.Data[0]
        #index_data['stock_code'] =symbol
        index_data['open'] = stock.Data[1]
        index_data['high'] = stock.Data[2]
        index_data['low'] = stock.Data[3]
        index_data['close'] = stock.Data[4]
        index_data['pre_close'] = stock.Data[5]
        index_data['volume'] = stock.Data[6]
        index_data['amt'] = stock.Data[7]
        index_data['dealnum'] = stock.Data[8]
        index_data['chg'] = stock.Data[9]
        index_data['pct_chg'] = stock.Data[10]
        # index_data['pct_chg']=index_data['pct_chg']/100
        index_data['vwap'] = stock.Data[11]
        index_data['adj_factor'] = stock.Data[12]
        index_data['close2'] = stock.Data[13]
        index_data['turn'] = stock.Data[14]
        index_data['free_turn'] = stock.Data[15]
        index_data['oi'] = stock.Data[16]
        index_data['oi_chg'] = stock.Data[17]
        index_data['pre_settle'] = stock.Data[18]
        index_data['settle'] = stock.Data[19]
        index_data['chg_settlement'] = stock.Data[20]
        index_data['pct_chg_settlement'] = stock.Data[21]
        index_data['lastradeday_s'] = stock.Data[22]
        index_data['last_trade_day'] = stock.Data[23]
        index_data['rel_ipo_chg'] = stock.Data[24]
        index_data['rel_ipo_pct_chg'] = stock.Data[25]
        index_data['susp_reason'] = stock.Data[26]
        index_data['close3'] = stock.Data[27]
        index_data['pe_ttm'] = stock.Data[28]
        index_data['val_pe_deducted_ttm'] = stock.Data[29]
        index_data['pe_lyr'] = stock.Data[30]
        index_data['pb_lf'] = stock.Data[31]
        index_data['ps_ttm'] = stock.Data[32]
        index_data['ps_lyr'] = stock.Data[33]
        index_data['dividendyield2'] = stock.Data[34]
        index_data['ev'] = stock.Data[35]
        index_data['mkt_cap_ard'] = stock.Data[36]
        index_data['pb_mrq'] = stock.Data[37]
        index_data['pcf_ocf_ttm'] = stock.Data[38]
        index_data['pcf_ncf_ttm'] = stock.Data[39]
        index_data['pcf_ocflyr'] = stock.Data[40]
        index_data['pcf_ncflyr'] = stock.Data[41]
        index_data['trade_status'] = stock.Data[42]
        index_data['data_source'] = 'Wind'
        index_data['created_date'] = time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        index_data['updated_date'] = time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        index_data = index_data[index_data['open'] > 0]

    except:
        print(getCurrentTime(), stock_symbol, ": Data Process Error!")


def save_current_data(current_symbol, current_data):
    current_data.to_excel('.\RH\\'+current_symbol+'.xlsx')


def main():
    stockCodes = w.wset("sectorconstituent",
                        "sectorid=a001010100000000;field=wind_code")
    stock_pool = stockCodes.Data[0]
    for symbol in stock_pool:
        start_date = '2017-11-01'
        end_date = '2018-03-12'
        get_stock(symbol, start_date, end_date)
        save_current_data(symbol, index_data)


if __name__ == '__main__':
    main()
