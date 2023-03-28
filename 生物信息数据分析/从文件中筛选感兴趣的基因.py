import os
import pandas as pd
import re
os.chdir("E:\scripts_input_R_python\python\Input\wt")
pattern = re.compile(r'(.+)_Gene_differential_expression')

def Read(arg1):
    Comp = pattern.match(arg1)
    data = pd.read_excel(arg1)
    result = data[data['gene_name']=='Fndc5']
    result.loc[:,'Comp'] = Comp.group(1)
    return result
files = os.listdir("E:\售后\文献解读\性别")

for file in files:
    out = Read(file)
    out.to_csv('my_csv.csv', mode='a')

