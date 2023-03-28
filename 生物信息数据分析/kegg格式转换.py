import pandas as pd
import os
os.chdir("/Users/yangpei/YangPei/File/Python")
Input_data = pd.read_excel('Input/kegg背景2.xlsx', sheet_name=0)

def keggfun(id_nummber):
    Index_list = []
    for key, value in enumerate(Input_data['gene_id']):
        if value == id_nummber:
            Index_list.append(key)
        else:
            pass
    Data = Input_data.loc[Index_list, ['gene_id', 'pathway']]
    dicts = {''.join(set(Data['gene_id'])): ';'.join(list(Data['pathway']))}
    output_handle = pd.Series(dicts)
    return output_handle



# 使用apply, 不能运行成功
unique_id = set(Input_data['gene_id'])
gene_id = pd.DataFrame(list(unique_id))
top10 = gene_id.head(10)
result = top10.apply(keggfun)
print(result)

"""
# 使用map，map只能用于series数据结构中
unique_id = set(Input_data['gene_id'])
gene_id = pd.Series(list(unique_id))
top10 = gene_id.head(10)
result = top10.map(keggfun)
print(result)

# 使用for循环
for temp in Input_data['gene_id']:
    result = keggfun(temp)
    result.to_csv('Output/output_kegg背景.csv', mode='a', header=False)
"""

