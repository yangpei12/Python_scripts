import pandas as pd

path = "E:\售后\李淑霞\\result.txt"
with open(path, 'r') as file_handle:
    ls = []
    for line in file_handle.readlines():
        new_line = line.strip('\n').split('\t')
        ls.append(new_line)

    df = pd.DataFrame(ls, columns=["query acc.ver", "subject acc.ver",
                                         "% identity", "alignment length",
                                         "mismatches", "gap opens",
                                         "q. start", "q. end", "s. start",
                                         "s. end", "evalue", "bit score"])
    df2 = df.drop_duplicates('query acc.ver')
    df2.to_excel("E:\售后\李淑霞\\result.xlsx" )






