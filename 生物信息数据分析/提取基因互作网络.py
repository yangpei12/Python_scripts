path = "E:\\Notepad\extract_specific_gene_network.txt"
out_path = "E:\\Notepad\extract_specific_gene_network_out.txt"
output = open(out_path, 'a')
source = ["ENT1", "mecr"]
target = ["ENT1", "mecr"]

with open(path) as file_handle:
    for line in file_handle.readlines():
        ln = line.strip('\n').split("\t")
        if ln[0] in source and ln[1] in target:
            lines = ln.append("\n")
            print(lines)
            #output.writelines(lines)
        else:
            pass
