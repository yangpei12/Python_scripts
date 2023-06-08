import os
import math

os.chdir(r'E:\售后')


class GenomeSplit:
    # 输入单个染色体,形式为“chr1”,参数更新后，self.chromosome_file = chromosome_file重新声明
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.chromosome_file = '{}.fa'.format(self.chromosome)

    # 按照染色体将序列进行整合成一条长序列
    def seq_integration(self, file):
        with open(file, 'r') as input1:
            for line in input1.readlines():
                if line.startswith('>'):
                    chrs = line.strip('>\n')
                    output_handle = open('{0}.fa'.format(chrs), 'a')
                    output_handle.write('>{0}\n'.format(chrs))
                else:
                    output_handle.write('{0}'.format(line.strip('\n')))

    # 查看整合后染色体的长度
    def seq_length(self):
        with open(self.chromosome_file, 'r') as input_handle:
            for line in input_handle.readlines():
                if not line.startswith('>'):
                    return len(line)

    # 按照固定长度拆分数据,length参数使用seq_length()的返回值
    def split_seq(self, step, length):
        frequency = math.ceil(length / step)
        output1 = open('{0}_split.fa'.format(self.chromosome_file), 'a')
        str_slice = map(lambda x: [step * x, step * (x + 1)], range(0, frequency))
        for tmp in str_slice:
            with open(self.chromosome_file, 'r') as input2:
                for line in input2.readlines():
                    if line.startswith('>'):
                        chrs = line.strip('\n')
                        output1.write('{0}_{1}:{2}\n'.format(chrs, tmp[0], tmp[1]))
                    else:
                        output1.write('{0}\n'.format(line[tmp[0]:tmp[1]]))


chr1_seq = GenomeSplit('chr1')
chr1_seq.seq_integration('genome.fa')
chr1_seq.seq_length()
chr1_seq.split_seq(40, chr1_seq.seq_length())
