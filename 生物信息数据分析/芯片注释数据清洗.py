import os
import pandas as pd
os.chdir("E:\售后\朱翠珍\GSE")
output_handle = open("GSE73129\GSE73129_anno.txt",'a')
with open('GSE73129\GSE73129_family.soft.txt','r',encoding='UTF-8') as input_file:
    for line in input_file.readlines():
        if not line.startswith(("#","!","^")):
            single_line = line.split("\t")
            if len(single_line) >2:
                result = '{0}'.format('\t'.join(single_line))
                output_handle.write(result)
        else:
            pass

