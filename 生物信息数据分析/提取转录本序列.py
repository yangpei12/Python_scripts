import os
from operator import itemgetter
import textwrap
path = "E:\售后\张永峰\summary\\3_1_transcript_expression\merged.fa"
out = "E:\scripts_input_R_python\python\\transcript_list_zyf.txt"
out2 = "E:\scripts_input_R_python\python\\transcript_list_zyf_out2.txt"
out_handle = open(out, 'a')
out_handle_2 = open(out2, 'a')
with open(path, 'r') as file_handle:
    for line in file_handle.readlines():
        if len(line) == 71:
            # print(line.strip('\n'))
            out_handle.write(line.strip('\n'))
        elif line.startswith('>'):
            # print(line.strip('\n'))
            out_handle.write(line.strip('\n') + '\t')
        else:
            # print(line)
            out_handle.write(line)

with open(out, 'r') as file_handle_2:
    dic = {}
    for line in file_handle_2.readlines():
        new_line = line.split('\t')
        dic[new_line[0]] = new_line[1]

    keys = ['>MSTRG.3.1', '>MSTRG.1.1', '>MSTRG.2.1', '>MSTRG.6.1']
    for k in keys:
        out_handle_2.write('{0} {1} {2}'.format(k, '\n', textwrap.fill(dic.get(k), 70))+'\n')
print('hello world')








