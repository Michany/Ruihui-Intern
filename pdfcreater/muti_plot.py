# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 21:52:49 2018

@author: 马海乾
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import re
import pylab as pl
import data_input
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from tictoc import tic, toc

import openpyxl
from openpyxl.styles import Font, numbers, Border, Side
from openpyxl.formatting.rule import DataBarRule
#用来转换excel到pdf
from win32com import client
#用来转换pdf到jpg
import ghostscript
from wand.image import Image

def plot_Y(df,tablename = 'stat_index_农商行_不良贷款与净息差' ,savesize=[85/25.4,50.4/25.4],figname='fig1',title = '', color=['#005DA6','#A7DBF7','#054E9F','#00A0E8','#003D75','#F8E824','#F08600'],significant=1):
    tic()
    #df = data_input.get_SOdata('StrategyOutput',tablename)
    #df.Date = [x.strftime("%Y-%m-%d") for x in df.Date]
    c_name = df.columns.values
    n_groups = df.shape[0]# 也可以用len(df.index)或者df[].count
    fig,ax = plt.subplots()
    opacity = 1 #透明度
    maxy = 0
    #miny = 0
    index = np.arange(n_groups)
    Xaxis = df.iloc[:,0]
    #获取bar列 
    bar_name=pd.DataFrame()
    for name in c_name:
        bar_compare = re.findall(r"b_(.+?)$",name)
        if bar_compare != []:
            bar_name[name]=bar_compare#正则匹配应画列       
    bar_num = bar_name.shape[1] #获取需要画bar列数
    #计算适当的bar的宽度
    bar_width = n_groups/((bar_num+1)*(n_groups+2))#bar数
    #获取line列
    line_name=pd.DataFrame()
    for name in c_name:
        line_compare = re.findall(r"l_(.+?)$",name)
        if line_compare != []:
            line_name[name]=line_compare#正则匹配应画列       
    line_num = line_name.shape[1] #获取需要画bar列数
  
    if bar_num>=1:            
        for i,key in enumerate(bar_name.columns):
            #print(i,key)
            if i == 0 :
                miny = min(df[key].values)
            else:
                miny = min(miny,min(df[key].values))
            maxy = max(maxy,max(df[key].values))
            ax.bar(index+i*bar_width, df[key].values, bar_width,
                        alpha=opacity, color=color[i],
                        yerr=None, label=bar_name[key].values[0])     
        
        for i,key in enumerate(line_name.columns):
            miny = min(miny,min(df[key].values))
            maxy = max(maxy,max(df[key].values))
            ax.plot(index, df[key].values,linewidth=3, color=color[bar_num+i],alpha=opacity,label=line_name[key].values[0])

        '''

        '''
    else:
        for i,key in enumerate(line_name.columns):
            if i == 0:
                miny = min(df[key].values)
            else:
                miny = min(miny,min(df[key].values))
            maxy = max(maxy,max(df[key].values))
            ax.plot(index, df[key].values, linewidth=3,color=color[bar_num+i],alpha=opacity,label=line_name[key].values[0])
        '''   
        if n_groups>15:
            ax.set_xticks(np.linspace(0,int((n_groups-1)/5)*5,5) + bar_width / 2)
            ax.set_xticklabels(df.iloc[np.linspace(0,int((n_groups-1)/5)*5,5),0].tolist())
        else:
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(df.iloc[:,0].tolist())
        '''
    if n_groups>15:
        ax.set_xticks(np.linspace(0,int((n_groups-1)/5)*5,5) + bar_width / 2)
        #ax.set_xticklabels(Xaxis.iloc[np.linspace(0,int((n_groups-1)/5)*5,5)].tolist())
        if Xaxis.dtype == 'datetime64[ns]' or Xaxis.dtype == 'datetime64' :
            ax.set_xticklabels([x.strftime("%Y-%m-%d") for x in Xaxis.iloc[np.linspace(0,int((n_groups-1)/5)*5,5)]])
    else:
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(Xaxis.iloc[:].tolist())
        if Xaxis.dtype == 'datetime64[ns]' or Xaxis.dtype == 'datetime64' :
            ax.set_xticklabels([x.strftime("%Y-%m-%d") for x in Xaxis.iloc[:]])
            
    ax.xaxis.set_ticks_position('none') #取消x轴刻度
    ax.yaxis.set_ticks_position('none') #取消y轴刻度 
    ax.legend(loc=9,ncol=line_num+bar_num,fontsize='medium',frameon=0) #设置图例   
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    if significant==0: 
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))#重要，y轴格式调整
    elif significant==1:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    elif significant==2:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    
    if bar_num != 0 and miny>=0:
        miny=0
        
    ax.set_ylim([miny-0.2*(maxy-miny), maxy+0.2*(maxy-miny)]) #设置y轴范围
    ax.set_yticks(np.linspace(miny-0.2*(maxy-miny), maxy+0.2*(maxy-miny),6)) #设置y轴标签
    pl.xticks(rotation=360)#用于旋转x轴标签
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
    ax.spines['bottom'].set_visible(False) #去掉上边框
    ax.spines['left'].set_visible(False) #去掉右边框
    ax.set_axisbelow(True)#关键！！！！使得grid置于图表下
    ax.grid(which='major', axis='y', linewidth=20, linestyle='-',color='#EEF8FD')#产生背底
    #print(bar_num)
    ax.set_title(title,fontdict = {'fontsize':15},loc='left',y=1.05)
    #ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    fig.set_size_inches(savesize[0]*2, savesize[1]*2)
    fig.savefig(r"./figures/%s.jpg"%figname, dpi=500,bbox_inches = 'tight')#用来去除白边，当不去True时
    #print(maxy,miny)
    toc()
    print(figname+' succeed!')
    
       
def plot_2Y(df,tablename = 'stat_index_农商行_不良贷款与净息差' ,savesize=[85/25.4,50.4/25.4],figname='fig1',title = '', color=['#A7DBF7','#054E9F','#00A0E8','#003D75','#005DA6','#F8E824','#F08600'],significant=1):
    tic()
   # df.Date = [x.strftime("%Y-%m-%d") for x in df.Date]
    n_groups = df.shape[0]# 也可以用len(df.index)或者df[].count 
    fig,ax = plt.subplots()# 列出图层
    ax2 = ax.twinx()#双y轴
    ls_all=[]
    opacity = 0.9 #透明度
    index = np.arange(n_groups)#生成x轴坐标
    df_bar_1Y = pd.DataFrame()
    df_lin_1Y = pd.DataFrame()
    df_bar_2Y = pd.DataFrame()
    df_lin_2Y = pd.DataFrame()
    
    for col in df.columns:
        col_split = re.split('_',col)
        if col_split[0] == 'b' and col_split[2] == '1':
            df_bar_1Y[col_split[1]] = df[col]
        elif col_split[0] == 'b' and col_split[2] == '2':
            df_bar_2Y[col_split[1]] = df[col]
        elif col_split[0] == 'l' and col_split[2] == '1':
            df_lin_1Y[col_split[1]] = df[col]
        elif col_split[0] == 'l' and col_split[2] == '2':
            df_lin_2Y[col_split[1]] = df[col]   
        elif col_split[0] != 'b' and col_split[0] != 'l':
            Xaxis =df[col]
    
    miny = np.nanmin([df_bar_1Y.min().min(),df_lin_1Y.min().min()]) 
    maxy = np.nanmax([df_bar_1Y.max().max(),df_lin_1Y.max().max()])  
    miny2 =np.nanmin([df_bar_2Y.min().min(),df_lin_2Y.min().min()]) 
    maxy2 = np.nanmax([df_bar_2Y.max().max(),df_lin_2Y.max().max()])  
    if n_groups>80:
        bar_width = 1.5#n_groups/((df_bar_1Y.shape[1]+df_bar_2Y.shape[1]+1)*n_groups)
    else:
        bar_width = n_groups/((df_bar_1Y.shape[1]+df_bar_2Y.shape[1]+1)*n_groups)
    #%%
    
    for i,row in enumerate(df_bar_1Y.columns):
        ls1 = ax.bar(index+i*bar_width, df_bar_1Y[row].values, bar_width,
                        alpha=opacity, color=color[i],
                        yerr=None, label=row)
        ls_all.append(ls1)
    
    for i,row in enumerate(df_bar_2Y.columns):
        ls2 = ax2.bar(index+(i+df_bar_1Y.shape[1])*bar_width, df_bar_2Y[row].values, bar_width,
                        alpha=opacity, color=color[i+df_bar_1Y.shape[1]],
                        yerr=None, label=row)
        ls_all.append(ls2)
    
    for i,row in enumerate(df_lin_1Y.columns):
        ls3, = ax.plot(index, df_lin_1Y[row].values,linewidth=3,
                      color=color[df_bar_1Y.shape[1]+df_bar_2Y.shape[1]+i],
                      alpha=opacity,label=row)
        ls_all.append(ls3)
        
    for i,row in enumerate(df_lin_2Y.columns): 
        ls4, = ax2.plot(index, df_lin_2Y[row].values,linewidth=3, #!!加逗号用于参数解包
                      color=color[df_bar_1Y.shape[1]+df_bar_2Y.shape[1]+df_lin_1Y.shape[1]+i],
                      alpha=opacity,label=row)
        ls_all.append(ls4)
    
    

    #%%
    #ax2.set_xticks([])
    
    if n_groups>15:
        ax.set_xticks(np.linspace(0,int((n_groups-1)/5)*5,5) + bar_width / 2)
        #ax.set_xticklabels(Xaxis.iloc[np.linspace(0,int((n_groups-1)/5)*5,5)].tolist())
        if Xaxis.dtype == 'datetime64[ns]' or Xaxis.dtype == 'datetime64' :
            ax.set_xticklabels([x.strftime("%Y-%m-%d") for x in Xaxis.iloc[np.linspace(0,int((n_groups-1)/5)*5,5)]])
    else:
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(Xaxis.iloc[:].tolist())
        if Xaxis.dtype == 'datetime64[ns]' or Xaxis.dtype == 'datetime64' :
            ax.set_xticklabels([x.strftime("%Y-%m-%d") for x in Xaxis.iloc[:]])
    
    ax.xaxis.set_ticks_position('none') #取消x轴刻度
    ax.yaxis.set_ticks_position('none') #取消y轴刻度 
    ax2.yaxis.set_ticks_position('none') #取消y轴刻度 
    
    
    '''
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#00FFCC')
    '''

    ax.set_xlabel('')
    ax.set_ylabel('')
    
    ####
    
    if df_bar_1Y.shape[1] != 0:
        miny=min(0,miny)
    #ax.set_ylim([miny-0.1*(maxy-miny), maxy+0.2*(maxy-miny)]) #设置y轴范围
    #ax.set_yticks(np.linspace(miny-0.2*(maxy-miny), maxy+0.2*(maxy-miny),6)) #设置y轴标签
    
    if miny == 0:
        ax.set_ylim([0, maxy+0.2*(maxy-miny)]) #设置y轴范围
        ax.set_yticks(np.linspace(0, maxy+0.2*(maxy-miny),6)) #设置y轴标签  
    else:
        ax.set_ylim([miny-0.1*(maxy-miny), maxy+0.2*(maxy-miny)]) #设置y轴范围
        ax.set_yticks(np.linspace(miny-0.2*(maxy-miny), maxy+0.2*(maxy-miny),6)) #设置y轴标签
    
    if df_bar_2Y.shape[1] != 0:
        miny2=min(0,miny2)
    #ax2.set_ylim([miny2-0.1*(maxy2-miny2), maxy2+0.2*(maxy2-miny2)]) #设置y轴范围
    #ax2.set_yticks(np.linspace(0.8*miny2,1.2*maxy2,6)) #设置y轴标签 
    if miny == 0:
        ax2.set_ylim([0, maxy2+0.2*(maxy2-miny2)]) #设置y轴范围
        ax2.set_yticks(np.linspace(0, maxy2+0.2*(maxy2-miny2),6)) #设置y轴标签  
    else:
        ax2.set_ylim([miny2-0.1*(maxy2-miny2), maxy2+0.2*(maxy2-miny2)]) #设置y轴范围
        ax2.set_yticks(np.linspace(miny2-0.2*(maxy2-miny2), maxy2+0.2*(maxy2-miny2),6)) #设置y轴标签
    ###
    #pl.xticks(rotation=360)#用于旋转x轴标签
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
    ax.spines['bottom'].set_visible(False) #去掉上边框
    ax.spines['left'].set_visible(False) #去掉右边框
    ax2.spines['top'].set_visible(False) #去掉上边框
    ax2.spines['right'].set_visible(False) #去掉右边框
    ax2.spines['bottom'].set_visible(False) #去掉上边框
    ax2.spines['left'].set_visible(False) #去掉右边框
    
    #print(miny,maxy,miny2,maxy2)
    ax.set_axisbelow(True)#关键！！！！使得grid置于图表下
    ax.grid(which='major', axis='y', linewidth=20, linestyle='-',color='#EEF8FD')#产生背底
    labs = list(df_bar_1Y.columns)+list(df_bar_2Y.columns)+list(df_lin_1Y.columns)+list(df_lin_2Y.columns)
    ax.legend(tuple(ls_all), tuple(labs), loc=9, ncol=df_bar_1Y.shape[1]+df_bar_2Y.shape[1]+df_lin_1Y.shape[1]+df_lin_2Y.shape[1], fontsize='medium',frameon=0) #设置图例  
    
    if significant==0: 
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))#重要，y轴格式调整
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))#重要，y轴格式调整
    elif significant==1:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))#重要，y轴格式调整
    elif significant==2:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))#重要，y轴格式调整
    ax.set_title(title,fontdict = {'fontsize':15},loc='left',y=1.05)
    #plt.title('Interesting',fontsize = 'medium',bbox=dict(edgecolor='blue', alpha=0.65 )) 
    #%
    #ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    fig.set_size_inches(savesize[0]*2, savesize[1]*2)

    fig.savefig(r"./figures/%s.jpg"%figname, dpi=400,bbox_inches = 'tight')#用来去除白边，当不去True时
    plt.show()
    toc()
    print(figname+' succeed!')


def plot_pie(df,savesize=[85/25.4,50.4/25.4],figname='fig1',title = '', color=['#005DA6','#A7DBF7','#054E9F','#00A0E8','#003D75','#F8E824','#F08600'],significant=1):
    new_col = []
    sizes = pd.DataFrame()
    for col in df.columns:
        col_split = re.split('_',col)
        if col_split[0] == 'p':
            new_col.append(col_split[1])
            sizes[col_split[1]] = df[col]
    #df.columns = new_col
    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)
    
    fig1, ax1 = plt.subplots(figsize = (6,3),subplot_kw = {'aspect':'equal'})
    sizes = sizes.values.tolist()[0]
    '''
    wedges, texts, autotexts = ax1.pie(sizes, labels=new_col, autopct=lambda pct: func(pct, data),
            shadow=True, startangle=90)
    '''
    wedges, texts, autotexts = ax1.pie(sizes, labels=new_col, autopct='%d%%',colors = color[0:len(new_col)],
        shadow=True, startangle=90)
    
    ax1.legend(wedges, new_col,
          title="图例",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    ax1.set_title(title,fontdict = {'fontsize':15},loc='left',y=1.05)
   # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9,top=0.9, wspace=None, hspace=None)
    fig1.set_size_inches(savesize[0]*2, savesize[1]*2)
    fig1.savefig(r"./figures/%s.jpg"%figname, dpi=400,bbox_inches = 'tight')#用来去除白边，当不去True时
    plt.show()
#%%
# 收益率曲线图
def plot_pnl(df,tablename,savesize=[90/25.4,135/25.4],figname='fig1',title = '',significant=1):
    def generate_profit_curve(ans: pd.DataFrame):
        fig = plt.figure()
        fig.set_size_inches(18, 12)
        try:
            ans.index = ans.index.astype("datetime64")
        except:
            pass#print("[Warning] Table {} does not set Index Columns Type as \"datetime64\", will set automatically.")
        try:
            if ('NAV' in ans.columns) or ('Nav' in ans.columns):
                pass
            elif ('Return' in ans.columns):
                ans['NAV'] = ans['Return']; ans['Daily_pnl'] = ans.Return.diff()
            elif ('net' in ans.columns):
                ans['NAV'] = ans['net']; ans['Daily_pnl'] = ans['PNL']
            elif ('wt' in ans.columns):
                ans['NAV'] = ans['wt']/ans['wt'][0]; ans['Daily_pnl']=ans['wt'].diff()
            else:
                ans['NAV'] = (ans['MarketValue'][0] + ans['Total_pnl']) / (ans['MarketValue'][0] + ans['Total_pnl'][0])
        except:
            raise Exception(tablename + ": Can't find \"NAV\", \"net\" or \"pnl\", Please specify in the source code...")
        ax = fig.add_subplot(211)
        ax.plot(ans.index, ans['NAV'], linewidth=2, label='净值')
        ax.fill_between(ans.index, ans['NAV'], y2=1,
                        where=(ans['NAV'] < ans['NAV'].shift(1)) |
                        ((ans['NAV'] > ans['NAV'].shift(-1)) &
                         (ans['NAV'] >= ans['NAV'].shift(1))),
                        facecolor='grey',
                        alpha=0.3)
        # 最大回撤区域标注
        ax.legend(fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=14)
        plt.grid()
    
        bx = fig.add_subplot(212)
        width = 1
        if len(ans)>40: width = 1.5
        bx.bar(ans.index, ans['Daily_pnl'].where(ans['Daily_pnl'] > 0),
               width, label='当日盈亏+', color='red', alpha=0.8)
        bx.bar(ans.index, ans['Daily_pnl'].where(ans['Daily_pnl'] < 0),
               width, label='当日盈亏-', color='green', alpha=0.8)
        bx.legend(fontsize=15)
        plt.xticks(fontsize=15)
        plt.grid()
        fig.savefig(r"./figures/%s.jpg"%figname, dpi=400,bbox_inches = 'tight')
    generate_profit_curve(df)
# 画自相关系数图
def plot_autocorrelation(df,tablename,savesize=[90/25.4,135/25.4],figname='fig2',title = '',significant=1):
    from PIL import Image
    fig = plt.figure()
    fig.set_size_inches(8, 3)
    cx = fig.add_subplot(111)
    plt.ylim((-0.3,0.3))
    pd.plotting.autocorrelation_plot(df['NAV'].pct_change().dropna(), c='r',ax=cx)\
        .get_figure().savefig(r"./figures/%s_AC.jpg"%figname, dpi=300)
    
    img=Image.open(r"./figures/%s_AC.jpg"%figname)  #打开图像
    
    box=(img.getbbox()[2]*0.01,img.getbbox()[3]*0.2,img.getbbox()[2]*0.99,img.getbbox()[3]*0.72)
    clip=img.crop(box)
    #clip.save(r"./figures/%s_AC.jpg"%figname)
    
def summarize(t: pd.DataFrame):
    global year_groups
    t['Date/Time'] = t.index

    yearly_drawdown = dict()
    yearly_return = dict() # 记录年收益率
    t['Year']=pd.Series([t['Date/Time'][i].year for i in range(len(t))], index = t.index)
    t_group=t.groupby('Year')
    year_groups=[t_group.get_group(i) for i in t_group.groups.keys()]
    for year_group in year_groups:
        max_draw_down, temp_max_value = 0, year_group['NAV'][0]
        start_date, end_date, current_start_date = 0, 0, 0
        continous = False # 是否连续
        for i in year_group.index:
            if temp_max_value < year_group['NAV'][i]:
                current_start_date = year_group['Date/Time'][i]
                temp_max_value = max(temp_max_value, year_group['NAV'][i])
                continous = False
            else:
                if max_draw_down>year_group['NAV'][i]/temp_max_value-1:
                    if not continous: 
                        continous = True
                    max_draw_down = year_group['NAV'][i]/temp_max_value-1
                else:
                    if continous:
                        continous = False
                        start_date = current_start_date
                        end_date = year_group['Date/Time'][i]
        year = year_group['Year'][i]
        yearly_drawdown[year] = max_draw_down, start_date, end_date
        try:
            yearly_return[year] = year_group['NAV'][i]/year_group['NAV'][0]-1
        except:
            yearly_return[year] = year_group['NAV'][i]/1-1
#####################################################################
    一年前 = t.index[-1]-pd.Timedelta(1,'Y')
    max_draw_down, temp_max_value = 0, t[一年前:]['NAV'][0]
    start_date, end_date, current_start_date = 0, 0, 0
    continous = False # 是否连续
    
    for i in t[一年前:].index:
        if temp_max_value < t['NAV'][i]:
            current_start_date = t['Date/Time'][i]
            temp_max_value = max(temp_max_value, t['NAV'][i])
            continous = False
        else:
            if max_draw_down>t['NAV'][i]/temp_max_value-1:
                if not continous: 
                    continous = True
                max_draw_down = t['NAV'][i]/temp_max_value-1            
            else:
                if continous:
                    continous = False
                    start_date = current_start_date
                    end_date = t['Date/Time'][i]
        # 统计收益率
        近一年return= t['NAV'][i]/t[一年前:]['NAV'][0]-1
    print("近一年return {:.2%}, 近一年回撤 {:.2%}".format(近一年return, max_draw_down))
#####################################################################
#####################################################################
    三月前 = t.index[-1]-pd.Timedelta(3,'M')
    max_draw_down, temp_max_value = 0, t[三月前:]['NAV'][0]
    start_date, end_date, current_start_date = 0, 0, 0
    continous = False # 是否连续
    for i in t[三月前:].index:
        if temp_max_value < t['NAV'][i]:
            current_start_date = t['Date/Time'][i]
            temp_max_value = max(temp_max_value, t['NAV'][i])
            continous = False
        else:
            if max_draw_down>t['NAV'][i]/temp_max_value-1:
                if not continous: 
                    continous = True
                max_draw_down = t['NAV'][i]/temp_max_value-1            
            else:
                if continous:
                    continous = False
                    start_date = current_start_date
                    end_date = t['Date/Time'][i]
        # 统计收益率
        近三月return= t['NAV'][i]/t[三月前:]['NAV'][0]-1
    print("近三月return {:.2%}, 近三月回撤 {:.2%}".format(近三月return,max_draw_down))
#####################################################################

    
    max_draw_down, temp_max_value = 0, t['NAV'][0]
    start_date, end_date, current_start_date = 0, 0, 0
    continous = False # 是否连续

    for i in t.index:
        if temp_max_value < t['NAV'][i]:
            current_start_date = t['Date/Time'][i]
            temp_max_value = max(temp_max_value, t['NAV'][i])
            continous = False
        else:
            if max_draw_down>t['NAV'][i]/temp_max_value-1:
                if not continous: 
                    continous = True
                max_draw_down = t['NAV'][i]/temp_max_value-1            
            else:
                if continous:
                    continous = False
                    start_date = current_start_date
                    end_date = t['Date/Time'][i]

    
    
    total_return = t['NAV'][i]/t['NAV'][0]-1
    return max_draw_down, start_date, end_date, total_return, pd.Series(yearly_return), yearly_drawdown
#%%
def write_yeild_and_drawdown(temp:tuple, tablename:str):
    print("Writing into excel file...")
    result = openpyxl.Workbook()
    table = result.active
    # 样式
    font = Font(name='fangsong', size=10)
    rule1 = DataBarRule(start_type='percentile', start_value=0, end_type='percentile', end_value=99,
                       color="FFFF0000", showValue="None", minLength=None, maxLength=60)
    rule2 = DataBarRule(start_type='percentile', start_value=90, end_type='percentile', end_value=0,
                       color="FF22ae6b", showValue="None", minLength=None, maxLength=60)
    format_number = numbers.BUILTIN_FORMATS[40]
    format_percent = numbers.BUILTIN_FORMATS[10]
    thin = Side(border_style="thin", color="000000")
    
    columns = ["","收益率","累计收益率","最大回撤","收益回撤比","最大回撤发生时间段"]
    table.append(columns)
    
    # 写入回撤
    yearly_drawdown = temp[5];i=0; ratio = dict()
    for year in yearly_drawdown.keys():
        table[2+i][3].value = yearly_drawdown[year][0]
        ratio[year] = yearly_drawdown[year][0]
        table[2+i][3].number_format = format_percent
        try:
            table[2+i][5].value = "{} - {}".format(yearly_drawdown[year][1].strftime("%Y.%m.%d"),yearly_drawdown[year][2].strftime("%Y.%m.%d"))
        except:
            table[2+i][5].value = "{} - {}".format(yearly_drawdown[year][1],yearly_drawdown[year][2])
        i+=1  
    # 写入收益
    yearly_return = temp[4]
    accumulated_return = (yearly_return+1).cumprod()-1
    i = 0
    for year in yearly_return.keys(): 
        during_string = "{} - {}".format(year_groups[i].index[0].strftime("%Y.%m"), year_groups[i].index[-1].strftime("%Y.%m"))
        table[2+i][0].value = during_string
        table[2+i][1].value = yearly_return[year] # 写入年收益
        ratio[year] = - yearly_return[year]/ratio[year]
        table[2+i][4].value = ratio[year] #写入收入回撤比
        table[2+i][4].number_format = format_number
        table[2+i][2].value = accumulated_return[year] # 写入累计收益
        table[2+i][1].number_format, table[2+i][2].number_format = format_percent, format_percent   
        i+=1       
        
    # 添加数据条
    for col in range(6):
        for row in range(len(year_groups)+1):
            table[row+1][col].font = font
    table.conditional_formatting.add('B{}:B{}'.format(2, i+1), rule1)
    table.conditional_formatting.add('D{}:D{}'.format(2, i+1), rule2)
    
    # 设置列宽
    table.column_dimensions['A'].width = 16
    table.column_dimensions['B'].width = 12
    table.column_dimensions['C'].width = 12
    table.column_dimensions['D'].width = 12
    table.column_dimensions['E'].width = 12
    table.column_dimensions['F'].width = 23 
    
    nrows = table.max_row
    for row in range(nrows, 0, -1):
            for j in range(len(table[row])):
                table[row][j].border = Border(top=thin, left=thin, right=thin, bottom = thin)
    print("save to",r"./excel/%s_OUTPUT.xlsx"%tablename)
    result.save(r"./excel/%s_OUTPUT.xlsx"%tablename)
    
    return nrows #返回excel表格的最大行数 
def convert_excel_to_pdf(tablename):
    xlApp = client.Dispatch("Excel.Application")
    path = os.getcwd()
    books = xlApp.Workbooks.Open(path + '\\excel\\' + tablename + '_OUTPUT.xlsx')
    try:
        books.SaveAs(path + '\\figures\\' + tablename + '_OUTPUT.pdf', FileFormat=57)
    except Exception as e:
        print("Failed to convert")
        print(str(e))
    finally:
        books.Close()
        xlApp.Quit()
def convert_pdf_to_jpg(tablename, nrows):
    print("Converting")
    path = os.getcwd()+'\\figures\\'
    filename = path + tablename + '_OUTPUT.pdf'
    img=Image(filename=filename,resolution=500)
    imgname=path + tablename + '_OUTPUT.jpeg'
    img.convert('jpeg').save(filename=imgname)
    
    import PIL
    img=PIL.Image.open(imgname)  #打开图像
    
    box=(360,490,3733,510+95*nrows) #根据nrows来剪裁图像
    clip=img.crop(box)
    clip.save(imgname)
    os.remove(filename) 
def muti_plot(tablename = 'core_stock_Hedge_300_pnl' ,columns = ['*'],time_range='no',savesize=[85/25.4,50.4/25.4],figname='fig1',title = '', color=['#005DA6','#A7DBF7','#054E9F','#00A0E8','#003D75','#F8E824','#F08600']):
    try:
        df = data_input.get_SOdata('StrategyOutput',tablename,columns,time_range)
    except:
        print("Can't find the Strategy Output in database..." )
        file_path = input("Please enter your file path, and make sure there're certain columns.\n")#raise Exception("取数据失败，请检查tablename_set中的表名。")
        df = pd.read_excel(file_path)
    print("已获取数据:\n", df.dtypes, '\n', "length:", len(df))
    if len(df)>0:
        if 'date' in df.columns:
            df.set_index('date',inplace=True)
        elif 'Date' in df.columns:
            df.set_index('Date',inplace=True)
        else:
            raise KeyError("Can't find date index!")
        plot_pnl(df,tablename,savesize,figname,title)
        plot_autocorrelation(df,tablename,savesize,figname,title)
        returns_and_drawdowns = summarize(df)
        nrows = write_yeild_and_drawdown(returns_and_drawdowns, tablename)
        convert_excel_to_pdf(tablename)
        convert_pdf_to_jpg(tablename, nrows)
        return nrows #返回excel图最大列数
    else:
        pass

   
if __name__ =='__main__':
    muti_plot(tablename='hell-core_stock_Hedge_300_pnl',time_range='no',savesize=[100/25.4 ,50.4/25.4],figname='fig1',title='hh',color=['#A7DBF7','#00A0E8','#005DA6','#F8BB24','#003D75','#F08600','#054E9F'])                  
        
        
                            
   
            
            
            
            
            
            
            
            
            
                            