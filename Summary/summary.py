# -*- coding: utf-8 -*-

import pandas as pd
import time


class Sum_class():
    '''
    一个包含了signal的统计信息的类
    --------------------------

    包含的属性如下

    name, volume, profit

    需要逐个查看signal对象时，可以使用

    >>> s.signal_class_list
    '''

    def __init__(self, input_name):
        self.name = input_name
        self.volume = 0
        self.profit = 0
        self.timeseries = dict()

    def check(self):
        for i in range(len(t)):
            is_active = t.loc[i]['类型'] in ['买入', '卖空']  # 属于主动买入或者卖空操作
            if t.loc[i]['Signal'] == self.name and is_active:
                self.volume += t.loc[i]['Shares/Ctrts/Units - Profit/Loss']
                self.profit += t.loc[i]['Net Profit - Cum Net Profit']
                self.timeseries[t.loc[i]['Date/Time']]=t.loc[i]['% Profit']


def init_signal_class():
    '''
    获取所有主动操作的signal names。

    依次初始化各个signal:Sum_class对象。
    '''
    global signal_class_list
    # 获取所有主动操作的signal names
    buy_signal_types = t[t['类型'] == '买入'].groupby(
        'Signal')  # 统计所有属于主动买入或者卖空操作的Signal类型
    short_signal_types = t[t['类型'] == '卖空'].groupby(
        'Signal')  # 统计所有属于主动买入或者卖空操作的Signal类型
    buy_signals_name_list = buy_signal_types.size()._index
    short_signals_name_list = short_signal_types.size()._index

    signal_class_list = []
    for signal in buy_signals_name_list:
        signal_class_list.append(Sum_class(signal))
    for signal in short_signals_name_list:
        signal_class_list.append(Sum_class(signal))
    return '初始化完成'


def start():
    t0 = time.time()
    print('正在初始化...')
    init_signal_class()
    print('初始化完成!')
    print('Signal name', ' '*8, 'Profit(￥)\t\tVolume\tP/V')
    for signal in signal_class_list:
        signal.check()
        print('[', signal.name, ']', ' '*(15-len(signal.name)),
              round(signal.profit, 2), '\t\t', signal.volume,
              '\t',round(signal.profit/signal.volume, 3))
    t1 = time.time()
    tpy = t1-t0
    print('Finished Successfully in %5.3f seconds.' % tpy)


def read_files(filepath):
    global t
    try:
        # 读取excel文件，sheet_name='交易列表',header=2（标题位于第三行）
        t = pd.read_excel(filepath, sheet_name='交易列表', header=2)
    except:
        print('找不到文件')


if __name__ == '__main__':
    read_files(r"C:\Users\70242\Desktop\500(2).xlsx")
    start()
    print(signal_class_list[3].timeseries)