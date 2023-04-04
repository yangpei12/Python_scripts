# -*- coding: utf-8 -*-
# @ Time    : 2023/4/4 23:23
# @ Author  : pyang
# @ Contact : 695819143@qq.com
# @ SoftWare    : PyCharm

import os

os.chdir(r'E:\售后')
check_point = 1061200

# 首先输出gtf文件的各个基因的区间位置，可根据位置，然后检测切分点是否位于基因内部
output_buffer = open('gene_pos_info.txt', 'a')
with open('gtf_150.txt', 'r') as input_file:
    for line in input_file.readlines():
        gene_info = line.split('\t')
        gene_start_pos = int(gene_info[3])
        gene_end_pos = int(gene_info[4])
        gene_range = list(range(gene_start_pos, gene_end_pos))
        gene_id = gene_info[8].split(';')[0]
        gene_feature = gene_info[2]
        if gene_feature == 'gene' or gene_feature == 'transcript':
            gene_pos_info = '{0}\t{1}\t{2}\n'.format(gene_info[0], gene_start_pos, gene_end_pos)
            output_buffer.write(gene_pos_info)

        if check_point in gene_range:
            print(gene_id)
            break
        else:
            gene_info[0] = '{0}_{1}:{2}'.format(gene_info[0], 0, check_point)
            print(gene_info)




