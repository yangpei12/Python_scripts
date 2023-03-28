import pandas as pd
import os
import sys
path = sys.argv[1]
os.chdir(path)
df = pd.DataFrame({'Sample':['Raw Reads','reads < 15nt after removing 3 adaptor','Mapped Reads']})
txt_path = os.path.join(path,'*.txt')
file_list = glob.glob(txt_path)
for file in file_list:
    if os.path.isfile(file):
        data = pd.read_table(file)
        df = pd.merge(df,data)
df.to_excel('stat_out.xlsx',index=False)
