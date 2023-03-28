import os
import pandas as pd
os.chdir("E:\scripts_input_R_python\python\Input")
data = pd.read_table("primer.txt",header=None)
drop_data = data.drop_duplicates([0])
drop_data.to_excel('primer2.xlsx',index=False)