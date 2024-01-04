import pandas as pd
import os
os.chdir('/mnt/d/售后/蓝芳仁')

# 背景：将NCBI的NCxxx的染色体格式转换为chr格式
mapping_chr = pd.read_excel('mapping_chr.xlsx')
def chr_trans(tissue):
    input_path = '%s/1_genes_fpkm_expression.xlsx'%(tissue)
    exp_data = pd.read_excel(input_path)
    data_merged =  pd.merge(exp_data, mapping_chr, how='left', on='chr')
    null_index = data_merged['mapping_chr'].isnull()
    data_merged['mapping_chr'][null_index] = data_merged['chr'][null_index]
    data_length = data_merged.shape[1]
    re_index = [0, -1]
    re_index.extend(list(range(2, data_length-1)))
    merged_data = data_merged.iloc[:, re_index]
    merged_data.to_excel('%s_genes_fpkm_expression.xlsx'%(tissue), index=False)
    return 'Done'

chr_trans('xiabu')