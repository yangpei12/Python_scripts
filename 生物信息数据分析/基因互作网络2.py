# 本文件用于在模块中提取指定基因的互作关系,例如"ENT1", "mecr"基因之间的互作关系
in_path = "E:\\Notepad\extract_specific_gene_network.txt"
out_path = "E:\\Notepad\extract_specific_gene_network_out2.txt"
output = open(out_path, 'a')
source = ["ENT1", "mecr"]
target = ["ENT1", "mecr"]

with open(in_path) as file_handle:
    for line in file_handle.readlines():
        ln = line.strip('\n').split("\t")
        if ln[0] in source and ln[1] in target:
            strs = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(ln[0], ln[1], ln[2], ln[3], ln[4], ln[5], "\n")
            output.write(strs)