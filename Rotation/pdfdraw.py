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


#%%主函数
def hello_pdf(strtest = '',title=''):
    rl_config.warnOnMissingFontGlyphs = 0
    pdfmetrics.registerFont(TTFont('song', r"./fonts/simsun.ttc"))
    pdfmetrics.registerFont(TTFont('fs', r"./fonts/simfang.ttf"))
    pdfmetrics.registerFont(TTFont('hei', r"./fonts/simhei.ttf"))
    pdfmetrics.registerFont(TTFont('yh', r"./fonts/msyh.ttf"))
    pdfmetrics.registerFont(TTFont('yh2', r"./fonts/msyhbd.ttf"))
    #设置字体：常规、斜体、粗体、粗斜体
    addMapping('cjk', 0, 0, 'song')    #normal
    addMapping('cjk', 0, 1, 'fs')    #italic
    addMapping('cjk', 1, 0, 'hei')    #bold
    addMapping('cjk', 10, 1, 'yh')    #italic and bold 
    
    p = canvas.Canvas('test2.pdf')#,pagesize=
    #默认(0, 0)点在左下角，此处把原点(0,0)向上和向右移动，后面的尺寸都是相对与此原点设置的
    #注意：移动原点时，向右向上为正，坐标系也是向右为+x，向上为+y
    #p.translate(0.5*inch, 0.5*inch) 
    
    #给出时间
    p.setFont('yh', 13.5) #设置字体
    p.setStrokeColorRGB(0/255, 160/255, 232/255) #改变线颜色#设置颜色，画笔色和填充色
    p.setFillColorRGB(0/255, 160/255, 232/255) #改变字体颜色与填充颜色
    timenow = str(datetime.today())
    timenow = timenow[0:4]+'年'+timenow[5:7]+'月'+timenow[8:10]+'日'
    p.drawString((210-59)/25.4*inch,(297-20)/25.4*inch,  timenow)
    
    #给出标题
    p.setFont('yh2', 28) #设置字体
    p.setStrokeColorRGB(0/255, 123/255, 198/255) #改变线颜色#设置颜色，画笔色和填充色
    p.setFillColorRGB(0/255, 123/255, 198/255) #改变字体颜色与填充颜色
    p.drawString(46.5/25.4*inch,(297-19)/25.4*inch,  title)

    #画一条线
    p.setStrokeColorRGB(0/255, 123/255, 198/255) #改变线颜色#设置颜色，画笔色和填充色
    p.setFillColorRGB(0/255, 123/255, 198/255)
    p.setLineWidth(1)
    p.line(46.5/25.4*inch,(297-23)/25.4*inch, (210-19)/25.4*inch, (297-23)/25.4*inch)
    '''
    draw_string(string,x,y,font = 'song',size = 14,color = 'red', anchor = 'middle')
    Pa.drawOn(p, 0*inch, (297-49.5)/25.4*inch)
    '''
    #画矩形 共三个
    p.setStrokeColorRGB(216/255, 216/255, 216/255) 
    p.setFillColorRGB(216/255, 216/255, 216/255) #设置背景颜色
    p.rect(0, (297-51.5)/25.4*inch, 210/25.4*inch, 22/25.4*inch, fill=1)
    '''
    p.setStrokeColorRGB(0/255, 123/255, 198/255) 
    p.setFillColorRGB(0/255, 123/255, 198/255) #设置背景颜色
    p.rect(15/25.4*inch, (297-25)/25.4*inch, 25/25.4*inch, 25/25.4*inch, fill=1)
    '''
    p.setStrokeColorRGB(0/255, 123/255, 198/255) 
    p.setFillColorRGB(0/255, 123/255, 198/255) #设置背景颜色
    p.rect(0, 0, 210/25.4*inch, 9/25.4*inch, fill=1)
    
    #右下角脚注
    p.setFont('yh2', 10) #设置字体
    p.setStrokeColorRGB(1, 1, 1) #改变线颜色#设置颜色，画笔色和填充色
    p.setFillColorRGB(1, 1, 1) #改变字体颜色与填充颜色
    p.drawString((210-42)/25.4*inch,3.5/25.4*inch,  u"数据来源：佰宽咨询")
    
    #文件头图片
    #p.drawImage(r"./figures/图例2.png", 21/25.4*inch, (297-19)/25.4*inch, (16/25.4)*inch, (16/25.4)*inch, mask=[0,2,40,42,136,139])
    p.drawImage(r"./图例2.png", 15/25.4*inch, (297-25)/25.4*inch, 25/25.4*inch, 25/25.4*inch)
    '''
    p.setStrokeColorRGB(0/255, 123/255, 198/255) 
    p.setFillColorRGB(0/255, 123/255, 198/255) #设置背景颜色
    p.rect(0, (297-49)/25.4*inch, 210/25.4*inch, 21/25.4*inch, fill=1)
    '''  
    #说明段落
    ParagraphStyle.defaults['wordWrap']="CJK" #实现中文自动换行
    styleSheet = getSampleStyleSheet()
    style = styleSheet['BodyText']
    style.fontName = 'yh'
    style.fontSize = 10 #字号 
    style.leading = 17 #设置行距
    style.leftIndent = 27/25.4*inch
    style.rightIndent = 0*27/25.4*inch
    style.textColor = colors.HexColor('#595757')#设置字体颜色80%灰
    #style.firstLineIndent = 32 #首行缩进
    #Pa = Paragraph(u'<b>这里是粗体</b>，<i>这里是斜体</i>, <strike>这是删除线</strike>, <u>这是下划线</u>, <sup>这是上标</sup>, <em>这里是强调</em>, <font color=#ff0000>这是红色</font>', style)
    Pa = Paragraph(strtest,style)
    Pa.wrapOn(p, 7*inch, 8*inch)
    Pa.drawOn(p, 0*inch, (297-49.5)/25.4*inch)
    
    #插入图片 五张
    p.drawImage(r"./fig1.jpg", 15/25.4*inch, 188/25.4*inch,187/25.4*inch, 45/25.4*inch)
    p.drawImage(r"./fig2.jpg", 15/25.4*inch, 121/25.4*inch, 187/25.4*inch, 56.3/25.4*inch)
    p.drawImage(r"./fig3.jpg", 15/25.4*inch, 66/25.4*inch, 187/25.4*inch, 45/25.4*inch)
    p.drawImage(r"./fig4.jpg", 15/25.4*inch, 12.5/25.4*inch, 187/25.4*inch, 45/25.4*inch)
    
    #表格
    '''
    tabl = table_model(data_table)
    tabl.wrapOn(p, 7*inch, 7.8*inch)
    tabl.drawOn(p, 0*inch, 7.5*inch)
    '''
    #柱状图1
    '''
    bar = autoLegender(draw_bar_chart(100, 300, ['a', 'b', 'c','d'], [(100, 200, 120, 300)],0,colors.HexColor("#7BB8E7"),125,150),'ruaruarua', 300, 250,'off')
    bar.wrapOn(p, 7*inch, 8*inch)
    bar.drawOn(p, 3.8*inch, 0*inch)
    bar = autoLegender(draw_bar_chart(100, 300, ['a', 'b', 'c','d'], [(100, 200, 120, 300)],0,colors.HexColor("#7BB8E7"),125,150),'ruaruarua', 300, 250,'off')
    bar.wrapOn(p, 0*inch, 0*inch)
    bar.drawOn(p, 0.4*inch, 0*inch)
    '''
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
    #保存pdf
    p.showPage()
    p.save()
    return 'succeed'


if __name__ == '__main__':
    hello_pdf()
   