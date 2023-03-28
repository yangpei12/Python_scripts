import os
import re
os.chdir("E:\scripts_input_R_python\python\\Input")
file_handle = open('gene_anno_100.txt')
file_handle2 = open('out_200.txt')
output = open('finall_out.txt','a')
dic = {}
for line in file_handle.readlines():
    line1 = line.strip('\n').split(' ')
    key = line1[0]
    value = '{0}{1}{2}'.format(line1[1]," ",line1[5])
    dic[key] = value

pattern = re.compile(r'(>GWHANUX[0-9]+):([0-9]+).([0-9]+)')
for line in file_handle2.readlines():
    line2 = line.strip('\n')
    result = pattern.match(line2)
    try:
        Id = result[0]
        Value = dic[Id]
        line3 = line2.replace(Id, Value)
        output.write('{0} {1}{2}'.format(Id, line3,'\n'))
    except TypeError:
        output.write('{0}{1}'.format(line2,'\n'))
    except KeyError:
        pass


