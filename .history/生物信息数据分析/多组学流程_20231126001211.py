import pandas as pd
import numpy as np
import scipy
import subprocess
import os
import multiprocessing

# 分析路径
itemPath = r'/Users/yangpei/YangPei/after_sale/test_1'
os.chdir(itemPath)

cond = 'HFDVSCD'

# =============================================================================
#                                        数据整理
# =============================================================================


# ========================== 1. 合并共同通路 ==========================
# 设置路径
mRNA_kegg_path = 'summary/{0}/alldata/KEGG_Enrichment.xlsx'.format(cond)
meta_kegg_path = 'summary/{0}/alldata/kegg.xlsx'.format(cond)


# 读取文件
mRNA_kegg_data = pd.read_excel(mRNA_kegg_path)
meta_kegg_data = pd.read_excel(meta_kegg_path)


# 选择通路并合并
mRNA_kegg_pathway_name = mRNA_kegg_data['Pathway_Name']
meta_kegg_pathway_name = meta_kegg_data['Pathway_Name']

common_pathway = set(mRNA_kegg_pathway_name) & set(meta_kegg_pathway_name)
common_pathway_df = pd.DataFrame({'Pathway_Name':list(common_pathway)})

# 输出共同同路表同时以供绘制图片
mRNA_kegg_pathway_selected_col = mRNA_kegg_data.loc[:, ['Pathway_Name', 'S', 'B', 'P.value']]
mRNA_kegg_pathway_selected_col.columns = ['Pathway_Name', 'S', 'B', 'Pvalue']
mRNA_common_kegg_pathway = pd.merge(common_pathway_df, mRNA_kegg_pathway_selected_col, on='Pathway_Name')
mRNA_common_kegg_pathway['type'] = 'mRNA'


meta_kegg_pathway_selected_col = meta_kegg_data.loc[:, ['Pathway_Name', 'S', 'B', 'P.value']]
meta_kegg_pathway_selected_col.rename(columns={'P.value': 'Pvalue'}, inplace=True)
meta_common_kegg_pathway = pd.merge(common_pathway_df, meta_kegg_pathway_selected_col, on='Pathway_Name')
meta_common_kegg_pathway['type'] = 'meta'

common_pathway_result = pd.concat([mRNA_common_kegg_pathway, meta_common_kegg_pathway])

output_name = 'summary/{0}/Integrative_Analysis/{0}_common_pathway_result.txt'.format(cond)
common_pathway_result.to_csv(output_name, sep='\t', index=False)


# ========================== 2. 差异基因和差异代谢物相关性分析 ==========================
diff_mRNA_path = 'summary/{0}/alldata/Gene_differential_expression.xlsx'.format(cond)
diff_meta_path = 'summary/{0}/alldata/significant.xlsx'.format(cond)

mRNA_exp = pd.read_excel(diff_mRNA_path, index_col=0)
diff_mRNA_exp = mRNA_exp.query('(significant=="yes")')
diff_mRNA_exp.columns = diff_mRNA_exp.columns.str.lstrip('FPKM.')
diff_meta_exp = pd.read_excel(diff_meta_path, index_col=4)

# 相关性分析的要求样本长度必须一致因此需要筛选共同的转录组样本与谢组样本。
common_sample = set(list(diff_mRNA_exp)) & set(list(diff_meta_exp))

# 由于集合是无序的，按照处理组的命名排序后
treat_group = cond.split('VS')[0]
sorted_samples = sorted(common_sample, key=lambda x: (x[:len(treat_group)] != treat_group, x))

# 对两组学的样本取完交集后并排序后，重新构建表达矩阵
diff_mRNA_exp_matrix = diff_mRNA_exp.loc[:, sorted_samples[:-1]]
diff_meta_exp_matrix = diff_meta_exp.loc[:, sorted_samples[:-1]]

# 根据表达矩阵，计算代谢物和基因的相关性，逐行扫描
def mRNA_extract_function(row_index):
    each_mRNA_exp = diff_mRNA_exp_matrix.iloc[row_index, :]
    mRNA_meta_corr = diff_meta_exp_matrix.apply(scipy.stats.pearsonr, axis=1, args=(each_mRNA_exp, ))
    mRNA_name = diff_mRNA_exp_matrix.index[row_index]
    meta_name = diff_meta_exp_matrix.index
    mRNA_meta_corr_dict = dict(mRNA_meta_corr.values)
    mRNA_meta_corr_df = pd.DataFrame({'mRNA':[mRNA_name]*len(meta_name),'metabolite':list(meta_name), 
                                        'corr_coef':list(mRNA_meta_corr_dict.keys()), 
                                        'pvalue':list(mRNA_meta_corr_dict.values())})
    return mRNA_meta_corr_df

diff_mRNA_nums = diff_mRNA_exp_matrix.shape[0]
output_name = 'summary/{0}/Integrative_Analysis/{0}_mRNA_meta_corr_result.txt'.format(cond, cond)

# 执行并行计算
if __name__ == '__main__':
    with multiprocessing.Pool(8) as p:

        # result为函数返回值的列表
        result  = p.map(mRNA_extract_function, list(range(diff_mRNA_nums)))
        mRNA_meta_corr_matrix = pd.concat(result)
        mRNA_meta_corr_matrix.to_csv(output_name, sep='\t', index=False)

# 差异表达矩阵
diff_mRNA_exp_matrix.to_csv('summary/{0}/Integrative_Analysis/{0}_mRNA_exp_matrix.txt'.format(cond, cond), sep='\t', index=False)
diff_meta_exp_matrix.to_csv('summary/{0}/Integrative_Analysis/{0}_meta_exp_matrix.txt'.format(cond, cond), sep='\t', index=False)


# =============================================================================
#                                        绘图
# =============================================================================

# ========================== 1. 通路标色 ==========================
# 调用sh_diff_exp1.sh中的KEGG_pathwayview.pl

# ========================== 2. 使用R绘制共同通路图 ==========================
# R文件路径
common_pathway_plot_cmd = 'Rscript common_pathway_scatterplot.R'
subprocess.run(common_pathway_plot_cmd, shell=True, capture_output=True, encoding='utf-8')

# ========================== 3. 相关系数热图 ==========================
# R文件路径
corrcoef_heatmap_cmd = 'Rscript corr_heatmap.R'
subprocess.run(corrcoef_heatmap_cmd, shell=True, capture_output=True, encoding='utf-8')

# ========================== 4. 九象限图 =============================
corrcoef_heatmap_cmd = 'Rscript nine_quadrant.R'
subprocess.run(corrcoef_heatmap_cmd, shell=True, capture_output=True, encoding='utf-8')


# ========================== 5. O2PLS分析 ============================
# R文件路径
o2pls_analysis_cmd = 'Rscript O2PLS_analysis.R'
subprocess.run(o2pls_analysis_cmd, shell=True, capture_output=True, encoding='utf-8')