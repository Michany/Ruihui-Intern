import pandas as pd
import matplotlib.pyplot as plt
from data_input import get_SOdata

SO_POOL = ["core_stock_Hedge_300_pnl", "core_stock_Hedge_500_L1_pnl", "core_stock_Hedge_500_L2_pnl", "core_stock_sp_qs1",
           "core_stock_TS_Hedge_300_pnl", "core_stock_TS_Hedge_500_pnl", "core_stock_TS_qs1_300_pnl", "core_stock_TS_qs1_500_pnl"]


def generate_profit_curve(ans: pd.DataFrame, symbol:str):
    fig = plt.figure()
    fig.set_size_inches(18, 12)

    ax = fig.add_subplot(211)
    ax.plot(ans.index, ans['Total_pnl'], linewidth=2, label='确认收益')
    ax.fill_between(ans.index, ans.Total_pnl, y2=0,
                    where=(ans.Total_pnl < ans.Total_pnl.shift(1)) |
                    ((ans.Total_pnl > ans.Total_pnl.shift(-1)) &
                     (ans.Total_pnl >= ans.Total_pnl.shift(1))),
                    facecolor='grey',
                    alpha=0.3)
    # 最大回撤标注
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax.legend(fontsize=15)
    plt.grid()

    bx = fig.add_subplot(212)
    width = 1
    if len(ans)>30: width = 1.5
    bx.bar(ans.index, ans.Daily_pnl.where(ans.Daily_pnl > 0),
           width, label='当日盈亏', color='red', alpha=0.8)
    bx.bar(ans.index, ans.Daily_pnl.where(ans.Daily_pnl < 0),
           width, label='当日盈亏', color='green', alpha=0.8)
    bx.legend(fontsize=15)
    plt.grid()
    fig.savefig(so+'.png', dpi=144, bbox_inches='tight')


for so in SO_POOL:
    temp = get_SOdata(so)
    generate_profit_curve(temp, so)
