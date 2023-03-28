import os
import re
os.chdir("E:\scripts_input_R_python\python")
pattern = re.compile(r'(\.[0-9]+)')
with open('uniprot_id_10.txt','r') as file_handle:
    for line in file_handle.readlines():
        new_line = line.strip('\n')
        #print(new_line)
        print(pattern.sub('',new_line))