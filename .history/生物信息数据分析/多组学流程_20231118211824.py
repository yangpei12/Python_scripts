import pandas as pd
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
diff_meta_path = 'meta/{0}/{1}_Gene_differential_expression.xlsx'.format(cond1, cond1)
