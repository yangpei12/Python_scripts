anno_path = "E:\售后\王秀玲\M155_CKVSWT_CK\M155_CKVSWT_CK_Gene_differential_expression.txt"
se_path = "E:\售后\王秀玲\M155_CKVSWT_CK\SE.MATS.JCEC.txt"
with open(anno_path, 'r') as handefile:
    for line in handefile.readlines():
        gene_name = line.split('\t')[0].split(";")
