# -*- coding: utf-8 -*-
# @ Time    : 2023/4/5 23:25
# @ Author  : pyang
# @ Contact : 695819143@qq.com
# @ SoftWare    : PyCharm
import os
os.chdir(r'E:\售后')

# 首先输出gtf文件的各个基因的区间位置，可根据位置，然后检测切分点是否位于基因内部
output_buffer = open('gene_pos_info.txt', 'a')
with open('gtf_150.txt', 'r') as input_file:
    for line in input_file.readlines():
        gene_infos = line.split('\t')
        gene_start = int(gene_infos[3])
        gene_end = int(gene_infos[4])
        gene_feature = gene_infos[2]
        if gene_infos[0] == 'chr1' and (gene_feature == 'gene' or gene_feature == 'transcript'):
            gene_pos_info = '{0}\t{1}\t{2}\n'.format(gene_infos[0], gene_start, gene_end)
            output_buffer.write(gene_pos_info)
output_buffer.close()