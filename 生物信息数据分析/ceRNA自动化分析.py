# -*- coding: utf-8 -*-
# @ Time    : 2023/6/27 22:53
# @ Author  : pyang
# @ Contact : 695819143@qq.com
# @ SoftWare    : PyCharm
import pandas as pd
import os

# ====更改工作路径====
os.chdir(r'E:\售后\ceRNA')
# ====step1: 读取数据====
# 读取ceRNA关系对
ceRNA_data = pd.read_csv('ceRNA.txt', sep='\t')

# ====筛选感兴趣的基因-读取基因==
target_rna_id = pd.read_csv('lncRNA.txt', sep='\t', header=None)

# ====step2: 读取注释信息数据 ====
# 读取mRNA注释信息
mRNA_anno = pd.read_csv('mRNA_anno.txt', sep='\t')
# 读取lncRNA注释信息
lncRNA_anno = pd.read_csv('lncRNA_anno.txt', sep='\t')
# 读取circRNA注释信息
circRNA_anno = pd.read_csv('circRNA_anno.txt', sep='\t')


class Analysis_ceRNA:
    def __init__(self, rna_class, target_rna_class):
        # rna_class：选择lncRNA还是circRNA
        # target_rna_class：感兴趣的RNA分了类型(lncRNA、circRNA、miRNA or mRNA)
        self.rna_class = rna_class
        self.target_rna_class = target_rna_class

    def select_ceRNA_network(self):
        # 在ceRNA关系对中筛选感兴趣的关系对lncRNA还是circRNA
        rna_ceRNA_pair_data = ceRNA_data[ceRNA_data['RNA_class'] == self.rna_class]

        # 使用rename按照需要修改指定列名，因此原始的ceRNA关系对中是RNA，为了后续merge，因此
        # 改为lncRNA
        rna_ceRNA_pair_data.rename(columns={'RNA': self.rna_class}, inplace=True)
        return rna_ceRNA_pair_data

    def select_target_rna(self):
        # 判断感兴趣的基因是否在mRNA列，返回值是布尔索引
        # target_rna_class参数是 'mRNA'
        rna_ceRNA_pair_data = Analysis_ceRNA.select_ceRNA_network(self)
        target_rna_bool = rna_ceRNA_pair_data[self.target_rna_class].isin(list(target_rna_id[0].values))
        target_rna_network = rna_ceRNA_pair_data[target_rna_bool]
        return target_rna_network

    def add_annotation_for_rna(self, gene_id_col, regulation_col):
        # 选择上下调信息，芯片项目和全转项目是不同的
        # 选择两列数据，一个是gene_id另外一个则是上下调
        gene_id = gene_id_col
        regulation = regulation_col
        mRNA_anno_regulation = mRNA_anno.loc[:, [gene_id, regulation]]
        lncRNA_anno_regulation = lncRNA_anno.loc[:, [gene_id, regulation]]
        circRNA_anno_regulation = circRNA_anno.loc[:, [gene_id, regulation]]

        # 对新添加的列进行重新命名，为的是后续筛选上下调方便
        mRNA_anno_regulation.columns = ['mRNA', 'mRNA_regulation']
        lncRNA_anno_regulation.columns = ['lncRNA', 'lncRNA_regulation']
        circRNA_anno_regulation.columns = ['circRNA', 'circRNA_regulation']

        target_rna_network = Analysis_ceRNA.select_target_rna(self)

        # 将感兴趣的关系对添加上下调信息
        length = len(target_rna_network)
        new_index = list(range(0, length))
        mRNA_regulation = pd.merge(target_rna_network, mRNA_anno_regulation, on='mRNA')
        lncRNA_regulation = pd.merge(target_rna_network, lncRNA_anno_regulation, on='lncRNA')

        # 重建索引至关重要，因为经过merge后mRNA_regulation与lncRNA_regulation这两个的数据框index发生了变化
        # 因此使用数据框的index属性，重新将所有的不同数据框修改成同一Index,这样concat能够正确合并
        mRNA_regulation_reindex = mRNA_regulation.drop_duplicates()
        mRNA_regulation_reindex.index = new_index
        lncRNA_regulation_reindex = lncRNA_regulation.drop_duplicates()
        lncRNA_regulation_reindex.index = new_index
        return mRNA_regulation_reindex, lncRNA_regulation_reindex

    def concatenate_data(self, gene_id_col, regulation_col):
        # 通过concat后会产生重复列，因此使用df.columns.duplicated()删除重复列名后，使用.loc选择子集
        # 在本函数中要想使用上一个函数的返回值，在本函数中的形参需要包含上一个函数的所有形参
        rna_regulation_data = Analysis_ceRNA.add_annotation_for_rna(self, gene_id_col, regulation_col)
        ceRNA_regulation = pd.concat([rna_regulation_data[0], rna_regulation_data[1]], axis=1)
        ceRNA_regulation = ceRNA_regulation.loc[:, ~ceRNA_regulation.columns.duplicated()]
        return ceRNA_regulation

a = Analysis_ceRNA('lncRNA', 'mRNA')
result = a.concatenate_data(gene_id_col='TargetID', regulation_col='regulation')
result.to_csv('result.csv')