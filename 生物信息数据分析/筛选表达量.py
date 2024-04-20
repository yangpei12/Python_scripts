import os
import pandas as pd
os.chdir(r'/Users/yangpei/YangPei/after_sale')
input_data = pd.read_excel('BVSA_total_lncRNA_differential_expression.xlsx', index_col=0)
# 按照分组进行切割
cond = 'BVSA'.split('VS')
treat = cond[0]
control = cond[1]
mRNA_exp_matrix = input_data.filter(like='FPKM.')
test_matrix = mRNA_exp_matrix.iloc[[0,1,2], :] 
# 首先判断是不是所有样本都小于阈值
gene_not_expression = test_matrix.iloc[2, :] < 1
if sum(gene_not_expression) == 6:
    pass

# 判断在阈值上的样本存在与对照组还是比较组, 判断在处理组或者对照组的TURE的数量即可
treat_exp = test_matrix.filter(like=treat).iloc[0, :]
control_exp = test_matrix.filter(like=control).iloc[0, :]

treat_condition = sum(treat_exp>1) >= 2
control_condition = sum(control_exp>1) >= 2
if treat_condition or control_condition:print('yes')