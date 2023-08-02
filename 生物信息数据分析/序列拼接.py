import difflib
import os
os.chdir(r'E:\售后')
d = difflib.Differ()
merged_seq = []

# 创建外部函数
def outter_fun(raw_seq):
# 创建内部函数
    def inter_fun():
        nonlocal raw_seq
        with open('test2.txt', 'r') as input_handle:
            for line in input_handle.readlines():
                if not line.startswith('>'):
                    query_seq = line.rstrip('\n')
                    dc = d.compare(raw_seq, query_seq)
                    merged_seq.extend([tmp for tmp in dc])
                    string1 = ''.join(merged_seq)
                    string2 = string1.replace('-','').replace('+','').replace(' ','')
                    raw_seq = string2 
        return raw_seq

    return inter_fun
f1 = outter_fun('CCTCTTTCCCTCGCAAGGACCGACAACGTGCGCAGCCTTCCCGCACGC')
f2 = f1()
print(f2)