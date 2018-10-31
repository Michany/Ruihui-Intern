# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:16:58 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-

#from django.http import HttpResponse
#from cStringIO import StringIO
#主干
from reportlab.pdfgen import canvas
from reportlab import rl_config
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib.fonts import addMapping
#实现中文自动换行
from reportlab.lib.styles import ParagraphStyle #ParagraphStyle.defaults['wordWrap']="CJK"
#画表
from reportlab.platypus import SimpleDocTemplate, Table,TableStyle#,Paragraph
#from reportlab.lib.units import inch
from reportlab.lib import colors
#画柱状图
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
#画框线
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.charts.legends import Legend
#画饼图
from reportlab.graphics.charts.piecharts import Pie
#文字
from  reportlab.graphics.shapes import String
#画折线图
from reportlab.graphics.charts.lineplots import LinePlot
#查看已经注册字体
from reportlab.pdfbase import pdfmetrics
#时间部分
from datetime import datetime
from muti_plot import muti_plot
from tictoc import *

#%%查看字体库
def view_fonts():
    print(pdfmetrics.getRegisteredFontNames())
#%%用于画一条线  
def draw_string(string,x,y,font = 'song',size = 14,color = 'red', anchor = 'middle'):
    s1 = String(x,y,string)
    s1.fontName = font
    s1.fontsize = size
    s1.fillColor = color.red
    s1.testAnchor = 'middle'
    return s1
#%% 用于给图表框，同时使其drawon
def autoLegender(chart, title = '', width = 448, height = 230, order = 'off',categories=[], use_colors=[]):
    d = Drawing(width,height)
    d.add(chart)
    lab = Label()
    lab.x = width/2  #x和y是title文字的位置坐标
    lab.y = 21/23*height
    lab.fontName = 'song' #增加对中文字体的支持
    lab.fontSize = 20
    lab.setText(title)
    d.add(lab)
    #颜色图例说明等
    if categories != [] and use_colors != []:
        leg = Legend()
        leg.x = 500   # 说明的x轴坐标 
        leg.y = 0     # 说明的y轴坐标
        leg.boxAnchor = 'se'
        # leg.strokeWidth = 4
        leg.strokeColor = None
        leg.subCols[1].align = 'right'
        leg.columnMaximum = 10  # 图例说明一列最多显示的个数
        leg.fontName = 'song' 
        leg.alignment = 'right' 
        leg.colorNamePairs = list(zip(use_colors, tuple(categories)))
        d.add(leg)
    if order == 'on':
        d.background = Rect(0,0,width,height,strokeWidth=1,strokeColor="#868686",fillColor=None) #边框颜色
    return d
#%%画表
def table_model(data,width = 7.1):
    #width = 7.1 总宽度
    colWidths = (width / len(data[0])) * inch   # 每列的宽度

    dis_list = []
    for x in data:
        # dis_list.append(map(lambda i: Paragraph('%s' % i, cn), x))
        dis_list.append(x)

    style = [
        # ('FONTNAME', (0, 0), (-1, -1), 'song'),  # 字体
        ('FONTSIZE', (0, 0), (-1, 0), 15),  # 字体大小
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d5dae6')),  # 设置第一行背景颜色
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#d5dae6')),  # 设置第二行背景颜色
        # 合并 （'SPAN',(第一个方格的左上角坐标)，(第二个方格的左上角坐标))，合并后的值为靠上一行的值，按照长方形合并
        ('SPAN',(0,0),(0,1)),
        ('SPAN',(1,0),(2,0)),
        ('SPAN',(3,0),(4,0)),
        ('SPAN',(5,0),(7,0)),

        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 对齐
        ('VALIGN', (-1, 0), (-2, 0), 'MIDDLE'),  # 对齐
        ('LINEBEFORE', (0, 0), (0, -1), 0.1, colors.grey),  # 设置表格左边线颜色为灰色，线宽为0.1
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.royalblue),  # 设置表格内文字颜色
        ('TEXTCOLOR', (0, -1), (-1, -1), colors.red),  # 设置表格内文字颜色
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # 设置表格框线为grey色，线宽为0.5
    ]
    component_table = Table(dis_list, colWidths=colWidths,style=style)

    return component_table
#%%画bar图
def draw_bar_chart(charmin, charmax, x_list, data=[()], x_label_angle=0, bar_color=colors.HexColor("#7BB8E7"), height=125, width=220):
    '''
    :param min: 设置y轴的最小值
    :param max: 设置y轴的最大值
    :param x_list: x轴上的标签
    :param data: y轴对应标签的值
    :param x_label_angle: x轴上标签的倾斜角度
    :param bar_color: 柱的颜色  可以是含有多种颜色的列表
    :param height: 柱状图的高度
    :param width: 柱状图的宽度
    :return: 
    '''
    bc = VerticalBarChart()
    bc.x = 50            # x和y是柱状图在框中的坐标
    bc.y = 50
    bc.height = height  # 柱状图的高度
    bc.width = width    # 柱状图的宽度
    bc.data = data      
    for j in range(0,len(x_list)):
        setattr(bc.bars[j], 'fillColor', bar_color)  # bar_color若含有多种颜色在这里分配bar_color[j]
    # 调整step
    minv = charmin * 0.5
    maxv = charmax * 1.5
    maxAxis = int(height/10)
    # 向上取整
    minStep = int((maxv-minv+maxAxis-1)/maxAxis)

    bc.valueAxis.valueMin = charmin * 0.5      #设置y轴的最小值
    bc.valueAxis.valueMax = charmax * 1.5      #设置y轴的最大值
    bc.valueAxis.valueStep = (charmax-charmin)/4   #设置y轴的最小度量单位
    if bc.valueAxis.valueStep < minStep:
        bc.valueAxis.valueStep = minStep
    if bc.valueAxis.valueStep == 0:
        bc.valueAxis.valueStep = 1
    bc.categoryAxis.labels.boxAnchor = 'ne'   # x轴下方标签坐标的开口方向
    bc.categoryAxis.labels.dx = -5           # x和y是x轴下方的标签距离x轴远近的坐标
    bc.categoryAxis.labels.dy = -5
    bc.categoryAxis.labels.angle = x_label_angle   # x轴上描述文字的倾斜角度
    # bc.categoryAxis.labels.fontName = 'song'
    x_real_list = []
    if len(x_list) > 10:
        for i in range(len(x_list)):
            tmp = '' if i%5 != 0 else x_list[i]
            x_real_list.append(tmp)
    else:
        x_real_list = x_list
    bc.categoryAxis.categoryNames = x_real_list
    return bc
#%%画饼图
def draw_pie(data=[], labels=[], use_colors=[], width=360,Height = 150):
    '''更多属性请查询reportlab.graphics.charts.piecharts.WedgeProperties'''

    pie = Pie()
    pie.x = 60 # x,y饼图在框中的坐标
    pie.y = 20
    pie.slices.label_boxStrokeColor = colors.white  #标签边框的颜色

    pie.data = data      # 饼图上的数据
    pie.labels = labels  # 数据的标签
    pie.simpleLabels = 0 # 0 标签在标注线的右侧；1 在线上边
    pie.sameRadii = 1    # 0 饼图是椭圆；1 饼图是圆形

    pie.slices.strokeColor = colors.red       # 圆饼的边界颜色
    pie.strokeWidth=1                         # 圆饼周围空白区域的宽度
    pie.strokeColor= colors.white             # 整体饼图边界的颜色
    pie.slices.label_pointer_piePad = 10       # 圆饼和标签的距离
    pie.slices.label_pointer_edgePad = 25    # 标签和外边框的距离
    pie.width = width
    pie.height = Height
    pie.direction = 'clockwise'
    pie.pointerLabelMode  = 'LeftRight'
    # for i in range(len(labels)):
    #     pie.slices[i].fontName = 'song' #设置中文
    for i, col in enumerate(use_colors):
         pie.slices[i].fillColor  = col
    return pie
#%%画折线图
def draw_lines(data,height=125, width=220,use_colors = []):
    lp = LinePlot()
    '''
    title = Label() 
    title.fontName  = "song"  
    title.fontSize  = 12  
    title_text = "ceshi"  
    title._text = title_text 
    '''
    lp.x = 50 #坐标轴中心坐标
    lp.y = 50
    lp.height = height
    lp.width = width
    thex = data.iloc[:,0].tolist()
    datalist = []
    for lines in range(len(data.columns)-1):
        datalist.append(list(zip(thex, data.iloc[:,lines+1].tolist())))
    lp.data = datalist
    if use_colors != []:      
        for i, col in enumerate(use_colors):
             lp.lines[i].strokeColor  = col          
    return lp

#%%主函数
def hello_pdf(strtest = '',title='', tablename = 'default', excel_rows = 3):
    rl_config.warnOnMissingFontGlyphs = 0
    pdfmetrics.registerFont(TTFont('song', r"./fonts/simsun.ttc"))
    pdfmetrics.registerFont(TTFont('fs', r"./fonts/simfang.ttf"))
    pdfmetrics.registerFont(TTFont('hei', r"./fonts/simhei.ttf"))
    pdfmetrics.registerFont(TTFont('yh', r"./fonts/msyh.ttf"))
    pdfmetrics.registerFont(TTFont('yh2', r"./fonts/msyhbd.ttf"))
    pdfmetrics.registerFont(TTFont('华文中宋', r"./fonts/STZHONGS.TTF"))
    pdfmetrics.registerFont(TTFont('颜体_准', r"./fonts/FZYanSJW_Zhun.ttf"))
    pdfmetrics.registerFont(TTFont('颜体_中', r"./fonts/FZYanSJW_Zhong.ttf"))
    #设置字体：常规、斜体、粗体、粗斜体
    addMapping('cjk', 0, 0, 'song')    #normal
    addMapping('cjk', 0, 1, 'fs')    #italic
    addMapping('cjk', 1, 0, 'hei')    #bold
    addMapping('cjk', 10, 1, 'yh')    #italic and bold 
    
    
    p = canvas.Canvas(tablename+'.pdf')#,pagesize=
    #默认(0, 0)点在左下角，此处把原点(0,0)向上和向右移动，后面的尺寸都是相对与此原点设置的
    #注意：移动原点时，向右向上为正，坐标系也是向右为+x，向上为+y
    #p.translate(0.5*inch, 0.5*inch) 

    
    '''
    #画一条线
    p.setStrokeColorRGB(0/255, 123/255, 198/255) #改变线颜色#设置颜色，画笔色和填充色
    p.setFillColorRGB(0/255, 123/255, 198/255)
    p.setLineWidth(1)
    p.line(46.5/25.4*inch,(297-23)/25.4*inch, (210-19)/25.4*inch, (297-23)/25.4*inch)
    
    draw_string(string,x,y,font = 'song',size = 14,color = 'red', anchor = 'middle')
    Pa.drawOn(p, 0*inch, (297-49.5)/25.4*inch)
    '''
    
    #画矩形 共三个
    '''p.setStrokeColorRGB(216/255, 216/255, 216/255) 
    p.setFillColorRGB(216/255, 216/255, 216/255) #设置背景颜色
    p.rect(0, (297-65)/25.4*inch, 210/25.4*inch, 22/25.4*inch, fill=1)

    p.setStrokeColorRGB(0/255, 123/255, 198/255) 
    p.setFillColorRGB(0/255, 123/255, 198/255) #设置背景颜色
    p.rect(0, 0, 210/25.4*inch, 9/25.4*inch, fill=1)
    '''
    #右上角“策略运行报告”
    p.drawImage(r"./figures/策略运行报告.png", 130/25.4*inch, 268/25.4*inch, 63/25.4*inch, 18/25.4*inch)

    #左上角公司Logo
    p.drawImage(r"./figures/Logo.png", 10/25.4*inch, 270/25.4*inch, 65/25.4*inch, 17/25.4*inch)

    #蓝色渐变底
    p.drawImage(r"./figures/back_block.png", 0/25.4*inch, 246/25.4*inch, 210/25.4*inch, 21/25.4*inch)
    #底部脚注的背景
    p.drawImage(r"./figures/back_block.png", 0/25.4*inch, 4/25.4*inch, 210/25.4*inch, 14/25.4*inch)


    #下脚注
    p.setFont('华文中宋', 14) #设置字体
    p.setStrokeColorRGB(38/255, 38/255, 38/255) #改变线颜色#设置颜色，画笔色和填充色
    p.setFillColorRGB(38/255, 38/255, 38/255) #改变字体颜色与填充颜色
    p.drawString(10/25.4*inch,11/25.4*inch,  u"专业实力 敏锐嗅觉 互利共赢")
    
    p.setFont('华文中宋', 8) #设置字体
    p.setStrokeColorRGB(38/255, 38/255, 38/255) #改变线颜色#设置颜色，画笔色和填充色
    p.setFillColorRGB(38/255, 38/255, 38/255) #改变字体颜色与填充颜色
    p.drawString(10/25.4*inch,7/25.4*inch,  u"免责声明：本资料中所有观点及投资组合运作表现仅供潜在投资人参考，并不构成管理人及投资顾问对投资者投资回报、经营业绩等任何承诺。")

    #给出标题
    p.setFont('颜体_中', 26) #设置字体
    p.setStrokeColorRGB(38/255, 38/255, 38/255) #改变线颜色#设置颜色，画笔色和填充色
    p.setFillColorRGB(38/255, 38/255, 38/255) #改变字体颜色与填充颜色
    p.drawString(65/25.4*inch,(297-62)/25.4*inch, title)
    
    '''
    p.setStrokeColorRGB(0/255, 123/255, 198/255) 
    p.setFillColorRGB(0/255, 123/255, 198/255) #设置背景颜色
    p.rect(0, (297-49)/25.4*inch, 210/25.4*inch, 21/25.4*inch, fill=1)
    '''  
    
    #说明段落
    ParagraphStyle.defaults['wordWrap']="CJK" #实现中文自动换行
    styleSheet = getSampleStyleSheet()
    style = styleSheet['BodyText']
    style.fontName = '华文中宋'
    style.fontSize = 11 #字号 
    style.leading = 17 #设置行距
    style.leftIndent = 15/25.4*inch
    style.rightIndent = 0*27/25.4*inch
    style.textColor = colors.HexColor('#111111')#设置字体颜色
    #style.firstLineIndent = 32 #首行缩进
    #Pa = Paragraph(u'<b>这里是粗体</b>，<i>这里是斜体</i>, <strike>这是删除线</strike>, <u>这是下划线</u>, <sup>这是上标</sup>, <em>这里是强调</em>, <font color=#ff0000>这是红色</font>', style)
    Pa = Paragraph(strtest,style)
    Pa.wrapOn(p, 7.6*inch, 10*inch)
    Pa.drawOn(p, 0*inch, (297-50)/25.4*inch)
    
    #插入小标题背景
    p.drawImage(r"./figures/title_block.png", 10/25.4*inch, (297-75-2)/25.4*inch, 
                80/25.4*inch, 8/25.4*inch)
    p.drawImage(r"./figures/title_block.png", 10/25.4*inch, (97+excel_rows*2)/25.4*inch, 
                80/25.4*inch, 8/25.4*inch)
    p.drawImage(r"./figures/title_block.png", 10/25.4*inch, 57/25.4*inch, 
                80/25.4*inch, 8/25.4*inch)
    
    #插入小标题(还要考虑年数很多的问题)
    p.setFont('颜体_中', 13) #设置字体
    p.setStrokeColorRGB(38/255, 38/255, 38/255) #改变线颜色#设置颜色，画笔色和填充色
    p.setFillColorRGB(38/255, 38/255, 38/255) #改变字体颜色与填充颜色
    p.drawString(15/25.4*inch,(297-74)/25.4*inch,  u"一、策略名字 策略历史收益")
    p.drawString(15/25.4*inch,(100+excel_rows*2)/25.4*inch,  u"二、年收益与波动")
    p.drawString(15/25.4*inch,60/25.4*inch,  u"三、收益率平稳性")
    
    #插入图片 3张
    p.drawImage(r"./figures/fig1.jpg", 35/25.4*inch, (297-95-80)/25.4*inch, 135/25.4*inch, 93/25.4*inch)
    p.drawImage(r"./figures/fig1_AC.jpg", 40/25.4*inch, 20/25.4*inch, 135/25.4*inch, 40/25.4*inch)
    p.drawImage(r"./figures/{}_OUTPUT.jpeg".format(tablename), 30/25.4*inch, (80-excel_rows)/25.4*inch, 150/25.4*inch, excel_rows*4.5/25.4*inch)
    
    #表格
    '''
    tabl = table_model(data_table)
    tabl.wrapOn(p, 7*inch, 7.8*inch)
    tabl.drawOn(p, 0*inch, 7.5*inch)
    '''
    #柱状图1

    #饼状图1
    '''
    pie = autoLegender(draw_pie(data_pie,labs_pie,color_pie,250),'hahaha',400,250,'off',labs_pie,color_pie)
    pie.wrapOn(p, 7*inch, 7.6*inch)
    pie.drawOn(p, 0*inch, 5.5*inch)
    '''
    #折线
    '''
    pl = autoLegender(draw_lines(data2,100,180,color_pie[0:6]),'',180,250,'off')
    pl.wrapOn(p, 7*inch, 8*inch)
    pl.drawOn(p, 0*inch, 3*inch)
    pl = autoLegender(draw_lines(data2,100,180,color_pie[0:6]),'',180,250,'off')
    pl.wrapOn(p, 7*inch, 8*inch)
    pl.drawOn(p, 3.5*inch, 3*inch)
    '''
    #加水印
    '''
    p.rotate(5)
    p.setFont('yh2', 60) #设置字体
    transparentblack = colors.Color(216/255,216/255,216/255,alpha = 0.2)
    p.setFillColor(transparentblack)
    p.drawString(40/24.5*inch, (297-100)/24.5*inch, u"智研")
    p.drawString(145/24.5*inch, (297-100)/24.5*inch, u"智研")
    
    p.drawString(36.66/24.5*inch, 141.34/24.5*inch, u"智研")
    p.drawString(141.66/24.5*inch, 141.34/24.5*inch, u"智研")
    
    p.drawString(33.33/24.5*inch, 70.67/24.5*inch, u"智研")
    p.drawString(138.33/24.5*inch, 70.67/24.5*inch, u"智研")
    
    p.drawString(30/24.5*inch, 15/24.5*inch, u"智研")
    p.drawString(135/24.5*inch, 15/24.5*inch, u"智研")
    transparentblack = colors.Color(0,0,0,alpha = 1)
    p.setFillColor(transparentblack)
    p.rotate(-5)  
    '''
    #保存pdf
    p.showPage()
    p.save()
    print("----- Successfully generated:",tablename, end='\n\n')
    return 'succeed'

#%%用于得到图片
def get_allImage(tablename = 'Default'):
    print("----- Generating:",tablename,)
    excel_rows = muti_plot(tablename=tablename)
    return excel_rows

def main(self, tablename_dict):
    for tablename in tablename_dict:
        print("----- Generating:",tablename,)
        excel_rows = get_allImage(tablename)
        hello_pdf(title='RSI横截面（纯多头）',strtest='锐汇资产团队介绍：基金经理和投资顾问具有高盛、美林等十年以上工作经验，以及拥有近20年的投资经验，广泛参与国内外资本市场运作。自主开发投资及研究自动化系统,从大数据中挖掘相关信息并提供相关咨询报告。',
                  tablename = tablename, excel_rows = excel_rows)
    return 
if __name__ == '__main__':
    import datetime
    #tablename_set = ['test0724','stat_index_股份制商业银行_不良贷款与净息差','test0724','stat_index_农商行_不良贷款与净息差','stat_index_2018年一季度不良率']
    tablename_set = ['RSI横截面_纯多头_sh300_日频_每年重置_'+datetime.date.today().strftime('%Y-%m-%d')]
    
    #title用来改标题
    #strtest用来改文本，每行用<br/>隔开
    for tablename in tablename_set:
        excel_rows = get_allImage(tablename)
        hello_pdf(title='RSI横截面（纯多头）500',
                  strtest='锐汇资产团队介绍：基金经理和投资顾问具有高盛、美林等十年以上工作经验，以及拥有近20年的投资经验，广泛参与国内外资本市场运作。自主开发投资及研究自动化系统,从大数据中挖掘相关信息并提供相关咨询报告。',
                  tablename = tablename, excel_rows = excel_rows)





           










