import os
import argparse
import pandas as pd

# 创建参数对象
parser = argparse.ArgumentParser(prog='RemoveGene',
                               description='This is a remove low expression gene scripts',
                               )

# 添加参数
parser.add_argument('inputFile', help='Please provide a input file path')
parser.add_argument('outputFile', help='Please provide a output file path')
parser.add_argument('cond', help='Please provide compare group')
parser.add_argument('fpkm_cut_off', help='Please provide fpkm cut off')
parser.add_argument('samples_cut_off', help='Please provide sample num cut off')

# 解析参数
args = parser.parse_args()

# 读取输入数据
input_data = pd.read_excel(args.inputFile)
# 按照分组进行切割
cond = args.cond.split('VS')
treat = cond[0]
control = cond[1]
mRNA_exp_matrix = input_data.filter(like='FPKM.')

row_ids = []
def filter_row(row_id, fpkm_cut_off, samples_cut_off):
    
    # 判断在阈值上的样本存在与对照组还是比较组, 判断在处理组或者对照组的TURE的数量即可
    treat_exp = mRNA_exp_matrix.filter(like=treat).iloc[row_id, :]
    control_exp = mRNA_exp_matrix.filter(like=control).iloc[row_id, :]

    treat_condition = sum(treat_exp > int(fpkm_cut_off)) >= int(samples_cut_off)
    control_condition = sum(control_exp > int(fpkm_cut_off)) >= int(samples_cut_off)

    if treat_condition or control_condition:
        row_ids.append(row_id) 
    else:
        pass

for id in range(mRNA_exp_matrix.shape[0]):
    filter_row(id, args.fpkm_cut_off, args.samples_cut_off)


filter_data = input_data.iloc[row_ids, :]
filter_data.to_excel(args.outputFile)
