import pandas as pd
import numpy as np
import re
import os
itemPath = r'E:\售后\多组学测试'
os.chdir(itemPath)

cond1 = 'CdOPVSCd'
cond2 = 'Cd+1_CdCK'


# ========================== 1. 合并共同通路 ==========================
# 设置路径
mRNA_kegg_path = 'mRNA/{0}/2_KEGG_Enrichment/{1}.KEGG_Enrichment.xlsx'.format(cond1, cond1)
meta_kegg_path = 'meta/{0}/idms2.pathway/{1}.xlsx'.format(cond2, cond2)


# 读取文件
mRNA_kegg_data = pd.read_excel(mRNA_kegg_path)
meta_kegg_data = pd.read_excel(meta_kegg_path)


# 选择通路并合并
mRNA_kegg_pathway_name = mRNA_kegg_data['Pathway_Name']
meta_kegg_pathway_name = meta_kegg_data['Pathway']

common_pathway = set(mRNA_kegg_pathway_name) & set(meta_kegg_pathway_name)
common_pathway_df = pd.DataFrame({'common_pathway':list(common_pathway)})

# ========================== 2. 差异基因和差异代谢物相关性 ==========================
diff_mRNA_path = 'mRNA/{0}/{1}_Gene_differential_expression.xlsx'.format(cond1, cond1)
diff_meta_path = 'meta/{0}/{1}.significant.idms2.xlsx'.format(cond2, cond2)

diff_mRNA_exp = pd.read_excel(diff_mRNA_path, index_col=0)
diff_meta_exp = pd.read_excel(diff_meta_path, index_col=0)

# 可根据sample_info整理成表达矩阵
diff_mRNA_exp_matrix = diff_mRNA_exp.filter(regex='FPKM.')
diff_meta_exp_matrix = diff_meta_exp.iloc[:,0:6]

# 根据表达矩阵，计算代谢物和基因的相关性，逐行扫描
def mRNA_extract_function(row_index):
    each_mRNA_exp = diff_mRNA_exp_matrix.iloc[row_index, :]

    def meta_extract_functioon():
        mRNA_meta_corr = diff_meta_exp_matrix.apply(np.corrcoef, axis=1, args=(each_mRNA_exp, ))
        return mRNA_meta_corr
    
    return meta_extract_functioon

diff_mRNA_exp_matrix.apply

pd.DataFrame