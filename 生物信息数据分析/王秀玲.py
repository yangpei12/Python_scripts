import pandas as pd
anno_path = "E:\售后\王秀玲\M155_CKVSWT_CK\M155_CKVSWT_CK_Gene_differential_expression.xlsx"
se_path = "E:\售后\王秀玲\M155_CKVSWT_CK\SE.MATS.JCEC.xlsx"
anno = pd.read_excel(anno_path)
se = pd.read_excel(se_path)
for gene in se.loc[:, 'geneSymbol']:
    an
    print(anno.str.findall(gene))


"""
for gene_name in se.loc[:,'geneSymbol']:
    
    if anno.loc[:, 'gene_name'].str.contains(gene_name):
        print('yes')
"""
