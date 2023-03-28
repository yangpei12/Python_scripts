# ceRNA数据整理
import os
import glob
import subprocess
import pandas as pd

def data_analysis(arg1):
    data = pd.read_table(arg1[0])
    output = data[data['significant'] == 'yes']
    output.loc[:, 't_id'] = output.loc[:, 't_name']
    return output

os.chdir(r"E:\售后\yd")
target_path = r"lncRNA\Output\COND1"
for roots,dirs,files in os.walk(target_path):
    for subdir in dirs:
        if subdir in os.listdir(target_path):
            path = os.path.join(roots,subdir,'mRNA_result','*_transcripts_annotation.txt')
            input_file = glob.glob(path)
            result = data_analysis(input_file)
            result.to_csv('{0}{1}'.format(subdir,'.csv'), index=False, mode='w')
            subprocess.run(['pwd'],shell=True)
            #subprocess.run(f"sed 's/,/\t/g' *.csv > .txt",shell=True)





