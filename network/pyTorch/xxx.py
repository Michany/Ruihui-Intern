# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:40:08 2018

@author: Administrator
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import re
import pylab as pl
import sys
sys.path.append(r'C:\Users\Administrator\Desktop\mhq\input_macrodata')
import data_input
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def plot_1Y(tablename = 'stat_macro_PB指数估值表', color=['#A7DBF7','#054E9F','#00A0E8','#005DA6','#003D75','#F8E824','#F08600'], title = '',savesize=[85/25.4,50.4/5]):
    df = data_input.get_SOdata(tablename)
    c_name = df.columns.values
    n_groups = df.shape[0]# 也可以用len(df.index)或者df[].count
    fig,ax = plt.subplots()
    opacity = 1 #透明度
    maxy = 0
    miny = 0
    index = np.arange(n_groups)
    #获取bar列
    bar_name={}
    for name in c_name:
        bar_compare = re.findall(r"b_(.+?)$",name)
        if bar_compare != []:
            bar_name[name]=bar_compare[0]#正则匹配应画列       
    bar_num = len(bar_name) #获取需要画bar列数
    #计算适当的bar的宽度
    bar_width = n_groups/((bar_num+1)*n_groups)#bar数
    #获取line列
    line_name={}
    for name in c_name:
        line_compare = re.findall(r"l_(.+?)$",name)
        if line_compare != []:
            line_name[name]=line_compare[0]#正则匹配应画列       
    line_num = len(bar_name) #获取需要画bar列数
  
    if bar_num>1:            
        for i,key in enumerate(bar_name.keys()):
            #print(i,key)
            miny = min(maxy,min(df[key].values))
            maxy = max(maxy,max(df[key].values))
            ax.bar(index+i*bar_width, df[key].values, bar_width,
                        alpha=opacity, color=color[i],
                        yerr=None, label=bar_name[key])     
        for i,key in enumerate(line_name.keys()):
            miny = min(maxy,min(df[key].values))
            maxy = max(maxy,max(df[key].values))
            ax.plot(index, df[key].values, color=color[bar_num+i],a