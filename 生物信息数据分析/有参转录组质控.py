# -*- coding: utf-8 -*-
# @ Time    : 2023/6/6 22:13
# @ Author  : pyang
# @ Contact : 695819143@qq.com
# @ SoftWare    :
import os
import pandas as pd
import subprocess
import sys
import re

# 输入工作路径
itemPath = r'E:\售后\refRNA'
os.chdir(itemPath)

# ========================== 1. 读取样本信息 ==========================
samples = pd.read_table('project_info/01_samples.txt')
projectInfo = samples.iloc[:, [1, 2]]
projectInfo.columns = ['文库名称', 'Sample']


# ========================== 2. 读取短片段和接头比例 ==========================
def cleanDataInfo(sample):
    dics = {}
    patternOne = re.compile(r'  Read 1 with adapter:\s+.+\s+(.+%)')
    patternTwo = re.compile(r'  Read 2 with adapter:\s+.+\s+(.+%)')
    patternThree = re.compile(r'Pairs that were too short:\s+.+\s+(.+%)')
    cleanDataPath = r'{0}\{1}\{2}_delete_adapter.summary'.format('CleanData', sample, sample)
    if os.path.exists(cleanDataPath):
        with open(cleanDataPath, 'r') as input_buffez:
            dics['Sample'] = sample
            for line in input_buffez.readlines():
                matchedOne = patternOne.search(line.strip('\n'))
                if matchedOne:
                    adapterOneRatio = matchedOne.group(1)
                    dics['Read1WithAdapter'] = [adapterOneRatio.strip('(')]

                matchedTwo = patternTwo.search(line.strip('\n'))
                if matchedTwo:
                    adapterTwoRatio = matchedTwo.group(1)
                    dics['Read2WithAdapter'] = [adapterTwoRatio.strip('(')]

                matchedThree = patternThree.search(line.strip('\n'))
                if matchedThree:
                    shortSeqRatio = matchedThree.group(1)
                    dics['PairsThatWereTooShort'] = [shortSeqRatio.strip('(')]
        df_merge = pd.DataFrame(dics)
        return df_merge
    else:
        df_merge = pd.DataFrame({'Sample': [sample], 'Read1WithAdapter': None,
                                 'Read2WithAdapter': None, 'PairsThatWereTooShort': None})
        return df_merge


# cleandata信息读取输出
sampleInfo = samples.iloc[:, 2]
cleanDataInfoResult = sampleInfo.map(cleanDataInfo)
cleanDataStat = pd.concat(cleanDataInfoResult.values)


# ========================== 3. 相关性读取代码 ==========================
def sampleCor(group_name):
    corData = pd.read_csv('Output/merged_result/correlation_cluster.txt', sep='\t', index_col=0)
    colData = corData.filter(like=group_name, axis=1)
    rowData = colData.filter(like=group_name, axis=0)

    # 按照样本筛选每个样本与其他样本的相关性
    sampleData = rowData.columns.map(lambda x: ';'.join(rowData.loc[:, x].values.astype(str)))
    sampleCorInfoResult = pd.DataFrame(sampleData, index=rowData.columns, columns=['CorrelationOfSample'])
    return sampleCorInfoResult


# 相关性输出
# samples.iloc[:,3]选取比较组列
if os.path.exists('Output/merged_result/correlation_cluster.txt'):
    groups = samples.iloc[:, 3].unique()  # 经过unique后数据类型转换为了array
    sampleCorResult = pd.Series(groups).map(sampleCor)  # 使用Series函数转换后才能使用map
    CorrelationData = pd.concat(sampleCorResult.values)
    CorrelationData['Sample'] = CorrelationData.index
else:
    CorrelationData = pd.DataFrame({'Sample': projectInfo.iloc[:, 1], 'CorrelationOfSample': None})


# ========================== 4. 读取stat_out文件 ====================================
def stat_out():
    if os.path.exists('Output/stat_out.txt'):
        statData = pd.read_csv('Output/stat_out.txt', sep='\t', header=0)
        newstatData = statData.iloc[1:, [0, 2, 4, 6, 7, 8]]
        newstatData.columns = ['Sample', 'RawData', 'CleanData', 'Q20', 'Q30', 'GC']
        return newstatData
    else:
        newstatData = pd.DataFrame({'Sample': projectInfo.iloc[:, 1], 'RawData': None, 'CleanData': None,
                                    'Q20': None, 'Q30': None, 'GC': None})
        return newstatData


# 输出stat_out文件
statOutData = stat_out()


# ========================== 5. 读取mapped_stat数据 ====================================
# 读取mapped_stat数据
def mapped_stat():
    if os.path.exists('Output/mapped_stat_out.txt'):
        statData = pd.read_csv('Output/mapped_stat_out.txt', sep='\t', header=0)
        selectData = statData.loc[:, ['Sample', 'Mapped reads', 'Unique Mapped reads', 'Multi Mapped reads']]
        selectData['Mappedreads'] = selectData['Mapped reads'].str.extract(r'([0-9]+.[0-9]+)%')
        selectData['UniqueMappedreads'] = selectData['Unique Mapped reads'].str.extract(r'([0-9]+.[0-9]+)%')
        selectData['MultiMappedreads'] = selectData['Multi Mapped reads'].str.extract(r'([0-9]+.[0-9]+)%')
        mapped_stat_result = selectData.loc[:, ['Sample', 'Mappedreads', 'UniqueMappedreads', 'MultiMappedreads']]
        return mapped_stat_result
    else:
        mapped_stat_result = pd.DataFrame(
            {'Sample': projectInfo.iloc[:, 1], 'Mappedreads': None, 'UniqueMappedreads': None,
             'MultiMappedreads': None})
        return mapped_stat_result


mappedStatData = mapped_stat()


# ========================== 6. 读取mapped_region数据 ====================================
def mapped_region():
    if os.path.exists('Output/mapped_region_stat.txt'):
        regionData = pd.read_csv('Output/mapped_region_stat.txt', sep='\t', header=0, index_col=0)
        regionDataTranspose = regionData.transpose()
        regionDataTranspose['Sample'] = regionDataTranspose.index
        return regionDataTranspose
    else:
        regionDataTranspose = pd.DataFrame(
            {'Sample': projectInfo.iloc[:, 1], 'exon': None, 'intron': None, 'intergenic': None})
        return regionDataTranspose


# 输出mapped_region
regionData = mapped_region()


# ========================== 7. 读取项目号及路径 ====================================
def item_info():
    if os.path.exists('project_info/04_report.txt'):
        reportData = pd.read_csv('project_info/04_report.txt', sep='\t', header=None, index_col=0)
        itemNum = reportData.loc['LCB_项目编号', 1]
        itemInfo = pd.DataFrame({'Sample': samples.iloc[:, 2], '项目编号': itemNum, '项目路径': itemPath})
        return itemInfo, itemNum
    else:
        itemInfo = pd.DataFrame({'Sample': samples.iloc[:, 2], '项目编号': None, '项目路径': None})
        return itemInfo


itemInfoData = item_info()[0]
itemNumber = item_info()[1]

# ========================== 将所有数据合并 ==========================
allDataOut = pd.DataFrame({'Sample': samples.iloc[:, 2]})
output_name = '{0}_check_data.txt'.format(itemNumber)
for tmp in [projectInfo, cleanDataStat, CorrelationData, statOutData,
            mappedStatData, regionData, itemInfoData]:
    allDataOut = pd.merge(allDataOut, tmp, on='Sample')
allDataOut.to_csv(output_name, sep='\t', index=False)
