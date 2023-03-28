import os
os.chdir(r'E:\售后')
out_file = open('a.fa','a')
s = 'GCTGATGTAAAGTAAGCTCGTGTGTCTACATCTAATCCTAC' \
    'TGTGAATATGTGGTGGGCTCATACAATAAAGCCTAGAAAGCC' \
    'AATAGACATTATTGCTCATACTATTCCTATATAGCCGAAAGG' \
    'TTCTTTTTTTCCGGAGTAGTAAGT'

# 50长度
def creative_seq(arg1):
    n = 1
    for i in range(0,arg1):
        info = '@A_'+ str(arg1) + '_'+ str(n)
        seq = s[0:(arg1-n)]+'T'* n
        info2 = '+'
        info3 = 'F'*arg1
        out_file.write('{0}\n{1}\n{2}\n{3}\n'.format(info,seq,info2,info3))
        n+=1
    return 'done'
result = creative_seq(150)






