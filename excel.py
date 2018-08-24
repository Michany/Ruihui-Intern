import xlrd
data = xlrd.open_workbook(r"C:\Users\70242\Desktop\500(2).xlsx")
table=data.sheets()[2]
nrows = table.nrows #行数
ncols = table.ncols #列数
'''for i in range(ncols):
    if table.col(i)[2].name=='交易列表':
        jiaoyiliebiao = i'''

def delete(original, pos):
    '''Inserts new inside original at pos.'''
    new = original[:pos]+ original[pos+1:]
    return new


def initial(sin):
    waiting = []
    count = 0
    for i in range(len(sin)):
        if sin[i] in ',¥':
            waiting.append(i)
    s=sin
    for i in waiting:
        s = delete(s, (i - count))
        count += 1
    return s

record=[]
for j in range(nrows):
    if table.col(7)[j].value.startswith('¥'):
        record.append(float(initial(table.col(7)[j].value)))
print(record)