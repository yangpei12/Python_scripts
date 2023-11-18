import pandas as pd
import os
itemPath = r'E:\售后\多组学测试'
os.chdir(itemPath)

cond1 = 'CdOPVSCd'
cond2 = 'Cd+1_CdCK'
mRNA_kegg_path = 'mRNA/{0}/2_KEGG_Enrichment/{1}.KEGG_Enrichment.xlsx'.format(cond1, cond1)
meta_kegg_path = 'meta/{0}/idms2.pathway/{1}.xlsx'.format(cond2, cond2)