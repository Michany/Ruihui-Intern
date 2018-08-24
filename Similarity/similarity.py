# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

#%% 获取历史行情数据
from data_reader import get_index_min
index = "399006.SZ"
start_date = "2000-01-01"
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
freq = "15min"
t = get_index_min(index, start_date, end_date, freq)

#%% 获取最近10个交易日（2周）的行情数据
start_date = (datetime.datetime.now() -
              datetime.timedelta(days=14)).strftime('%Y-%m-%d')
now = get_index_min(index, start_date, end_date, freq)

#%%
len_t, len_now = len(t), len(now)
max_corr, max_index = 0, -1
corr = list()
for i in range(0, len_t-len_now):
    temp_corr = np.corrcoef(x=t[i:i+len_now]['ssclose'], y=now['ssclose'])
    corr.append(temp_corr[1])
corr = pd.DataFrame(corr)
ans = corr[corr[0] > 0.85]
t=t.ssclose

group_list=[]; group=0; last_i=0
for i in ans.index:
    if i-last_i != 1: 
        group += 1
        last_i = i
    else:
        last_i += 1
    group_list.append(group)
ans = pd.DataFrame({0:list(ans[0]),1:group_list}, index=ans.index)
ans_groups = ans.groupby(1)

#%% 计算概率
forward_length = 960
index_max_corr = [int(ans_groups.get_group(j+1).idxmax()[0]) for j in range(ans_groups.size().size)]
prob_up = sum(t[i]<t[i+len_now+forward_length] for i in index_max_corr) / ans_groups.size().size
prob_down = 1-prob_up

if prob_up>prob_down:
    for i in index_max_corr:
        if t[i]<t[i+len_now+forward_length]:
            ans.loc[i, 'more_likely'] = 1
else:
    for i in index_max_corr:
        if t[i]>t[i+len_now+forward_length]:
            ans.loc[i, 'more_likely'] = 1

best_history_index = ans.where(ans.more_likely!=0).idxmax()[0]

#%% 调整坐标及缩放
i=best_history_index

future=dict()
future['upper']=t[i:i+len_now+forward_length].max()
future['lower']=t[i:i+len_now+forward_length].min()
future['mid']=(future['upper']+future['lower'])/2
future['width']=future['upper']-future['lower']
future['average']=t[i:i+len_now+forward_length].mean()
future['upward']=future['upper']-future['average']
future['downward']=future['average']-future['lower']

curt=dict()
curt['upper']=now['sclose'].max()
curt['lower']=now['sclose'].min()
curt['average']=now['sclose'].mean()
curt['width']=curt['upper']-curt['lower']
curt['mid']=(curt['upper']+curt['lower'])/2

a = (curt['lower']-curt['width']*future['downward']/future['width']*3)*future['average']/curt['average']
b = (curt['upper']+curt['width']*future['upward']/future['width']*3)*future['average']/curt['average']

#%% 画图
fig = plt.figure()
fig.set_size_inches(13, 9)
ax = fig.add_subplot(111)
ax.plot(
    range(len_now), now['sclose'], linewidth=3, label='current', c='#007ac6')
plt.ylim((curt['mid'] - future['downward'] * 3, curt['mid'] + future['upward'] * 3))
plt.grid(color='#007ac6', alpha=0.7, axis='y',linestyle = 'dashed')
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

ax2 = ax.twinx()
history = t[best_history_index:best_history_index + len_now + forward_length]
ax2.plot(
    range(len_now + forward_length),
    history,
    linewidth=3,
    label='history',
    c='#f08600')
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.ylim((a, b))
plt.axis('off')
plt.show()