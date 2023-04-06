# -*- coding: utf-8 -*-
# @ Time    : 2023/4/4 23:23
# @ Author  : pyang
# @ Contact : 695819143@qq.com
# @ SoftWare    : PyCharm

import os
import math

os.chdir(r'E:\售后')

class GtfSplit:
    # gtf_file:gtf文件名字，chromosome：染色体号
    def __init__(self, gtf_file, step, length):
        self.gtf_file = gtf_file
        self.step = step
        self.length = length

    # 读取gtf文件，按照染色体号将基因区间打印出来
    def gene_pos_info(self, output_file, chromosome):
        gtf_buffer = open(output_file,'a')
        # 将注释文件中的位置先打印出来，可根据每个基因的位置进行切割位点的设定
        with open(self.gtf_file, 'r') as input_file:
            for lines in input_file.readlines():
                gene_info = lines.split('\t')
                gene_start_pos = int(gene_info[3])
                gene_end_pos = int(gene_info[4])
                if gene_info[0] == chromosome:
                    frequency = math.ceil(self.length / self.step)
                    str_slice = map(lambda x: [self.step * x, self.step * (x + 1)], range(0, frequency))
                    for tmp in str_slice:
                        if gene_start_pos >= tmp[0] and gene_end_pos <= tmp[1]:
                            gene_info[0] = '{0}_{1}:{2}'.format(gene_info[0], tmp[0], tmp[1])
                            gtf_buffer.writelines(gene_info)


if __name__ == "__main__":
    gtf = GtfSplit('gtf_150.txt', 800000, 1100000)
    gtf.gene_pos_info('genome_splitd.gtf', 'chr1')
