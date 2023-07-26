# -*- coding: utf-8 -*-
# @ Time    : 2023/6/6 22:13
# @ Author  : pyang
# @ Contact : 695819143@qq.com
# @ SoftWare    :
import os
import argparse
import pandas as pd
import subprocess
import sys
import re
# ========================== 创建命令行参数 =========================
# 创建argparse对象
parser = argparse.ArgumentParser(
                    prog='This Program Is Checking Data Quality',
                    description='What the program does',
                    epilog='Text at the bottom of help')


# 输入工作路径
itemPath = sys.argv[1]
os.chdir(itemPath)

# ========================== 1. 读取样本及文库信息 ==========================
# 读取样本信息
sample_info =  pd.read_csv('{0}/sample_info.txt'.format(itemPath),sep='\t').drop_duplicates()
sample_info.rename(columns = {'#SampleID':'Sample','COND1':'COND1'}, inplace=True)

# 读取文库信息
samples = pd.read_csv('{0}/project_info/01_samples.txt'.format(itemPath),sep='\t').drop_duplicates()
projectInfo = samples.iloc[:, [1, 2]]
projectInfo.columns = ['文库名称', 'Sample']
libInfo = pd.merge(sample_info, projectInfo, on='Sample')
if libInfo.empty:
    libInfo = pd.DataFrame({'Sample': sample_info.iloc[:,0], '文库': None})
else:
    libInfo = libInfo.iloc[:,[0,2]]
    
# ========================== 2. 读取短片段和接头比例 ==========================
def cleanDataInfo(sample):
    dics = {}
    patternOne = re.compile(r'  Read 1 with adapter:\s+.+\s+(.+%)')
    patternTwo = re.compile(r'  Read 2 with adapter:\s+.+\s+(.+%)')
    patternThree = re.compile(r'Pairs that were too short:\s+.+\s+(.+%)')
    cleanDataPath = r'{0}/{1}/{2}/{3}_delete_adapter.summary'.format(itemPath,'CleanData', sample, sample)
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
sampleInfo = sample_info.iloc[:,0]
cleanDataInfoResult = sampleInfo.map(cleanDataInfo)
cleanDataStat = pd.concat(cleanDataInfoResult.values)

# ========================== 3. 相关性读取代码 ==========================
def sampleCor(group_name):
    corData = pd.read_csv('Output/merged_result/correlation_cluster.txt', sep='\t', index_col=0)
    group_name_bool = sample_info.iloc[:,1] == group_name # 筛选组名所在的行，返回布尔值
    sample_index = sample_info.iloc[:, 0][group_name_bool] # 根据组名返回的布尔值，筛选与组名对应的样本名
    rowData = corData.loc[list(sample_index.values),list(sample_index.values)]

    # 按照样本筛选每个样本与其他样本的相关性
    sampleData = rowData.columns.map(lambda x: ';'.join(rowData.loc[:, x].values.astype(str)))
    sampleCorInfoResult = pd.DataFrame(sampleData, index=rowData.columns, columns=['CorrelationOfSample'])
    return sampleCorInfoResult


# 相关性输出
# samples.iloc[:,3]选取比较组列
if os.path.exists('Output/merged_result/correlation_cluster.txt'):
    groups = sample_info.iloc[:,1].unique()  # 经过unique后数据类型转换为了array
    sampleCorResult = pd.Series(groups).map(sampleCor)  # 使用Series函数转换后才能使用map
    CorrelationData = pd.concat(sampleCorResult.values)
    CorrelationData['Sample'] = CorrelationData.index
else:
    CorrelationData = pd.DataFrame({'Sample': sample_info.iloc[:,0], 'CorrelationOfSample': None})

# ========================== 4. 读取stat_out文件 ====================================
def stat_out():
    if os.path.exists('{0}/Output/stat_out.txt'.format(itemPath)):
        statData = pd.read_csv('{0}/Output/stat_out.txt'.format(itemPath), sep='\t', header=0)
        newstatData = statData.iloc[1:, [0, 2, 4, 6, 7, 8]]
        newstatData.columns = ['Sample', 'RawData', 'CleanData', 'Q20', 'Q30', 'GC']
        return newstatData
    else:
        newstatData = pd.DataFrame({'Sample': sample_info.iloc[:,0], 'RawData': None, 'CleanData': None,
                                    'Q20': None, 'Q30': None, 'GC': None})
        return newstatData


# 输出stat_out文件
statOutData = stat_out()

# ========================== 5. 读取mapped_stat数据 ====================================
# 读取mapped_stat数据
def mapped_stat():
    if os.path.exists('{0}/Output/mapped_stat_out.txt'.format(itemPath)):
        statData = pd.read_csv('{0}/Output/mapped_stat_out.txt'.format(itemPath), sep='\t', header=0)
        selectData = statData.loc[:, ['Sample', 'Mapped reads', 'Unique Mapped reads', 'Multi Mapped reads']]
        selectData['Mappedreads'] = selectData['Mapped reads'].str.extract(r'([0-9]+.[0-9]+)%')
        selectData['UniqueMappedreads'] = selectData['Unique Mapped reads'].str.extract(r'([0-9]+.[0-9]+)%')
        selectData['MultiMappedreads'] = selectData['Multi Mapped reads'].str.extract(r'([0-9]+.[0-9]+)%')
        mapped_stat_result = selectData.loc[:, ['Sample', 'Mappedreads', 'UniqueMappedreads', 'MultiMappedreads']]
        return mapped_stat_result
    else:
        mapped_stat_result = pd.DataFrame(
            {'Sample': sample_info.iloc[:,0], 'Mappedreads': None, 'UniqueMappedreads': None,
             'MultiMappedreads': None})
        return mapped_stat_result

# 输出mapped_stat数据
mappedStatData = mapped_stat()

# ========================== 6. 读取mapped_region数据 ====================================
def mapped_region():
    if os.path.exists('{0}/Output/mapped_region_stat.txt'.format(itemPath)):
        regionData = pd.read_csv('{0}/Output/mapped_region_stat.txt'.format(itemPath), sep='\t', header=0, index_col=0)
        regionDataTranspose = regionData.transpose()
        regionDataTranspose['Sample'] = regionDataTranspose.index
        return regionDataTranspose
    else:
        regionDataTranspose = pd.DataFrame(
            {'Sample': sample_info.iloc[:,0], 'exon': None, 'intron': None, 'intergenic': None})
        return regionDataTranspose


# 输出mapped_region
regionData = mapped_region()

# ========================== 7. 读取项目号及路径 ====================================
def item_info():
    if os.path.exists('{0}/project_info/04_report.txt'.format(itemPath)):
        reportData = pd.read_csv('{0}/project_info/04_report.txt'.format(itemPath), sep='\t', header=None, index_col=0)
        itemNum = reportData.loc['LCB_项目编号', 1]
        itemInfo = pd.DataFrame({'Sample': sample_info.iloc[:,0], '项目编号': itemNum, '项目路径': itemPath})
        return itemInfo, itemNum
    else:
        itemInfo = pd.DataFrame({'Sample': sample_info.iloc[:,0], '项目编号': None, '项目路径': None})
        return itemInfo


itemInfoData = item_info()[0]
itemNumber = item_info()[1]

# ========================== 将所有数据合并 ==========================
allDataOut = pd.DataFrame({'Sample': sample_info.iloc[:,0]})
output_name = '/public23/4_After_Sales_Personal/YangPei/20.Script/Python_script/refRNA_stat/test/{0}_data_stat.txt'.format(itemNumber)
for tmp in [libInfo, cleanDataStat, CorrelationData, statOutData,
            mappedStatData, regionData, itemInfoData]:
    allDataOut = pd.merge(allDataOut, tmp, on='Sample')
allDataOut.to_csv(output_name, sep='\t', index=False)

# ========================= 异常比例统计 ===========================
# 定义各项目阈值
read1_with_adapter_cutoff = 10
read2_with_adapter_cutoff = 10
pairs_that_were_too_short_cutoff = 10
raw_data_cutoff = 6
clean_data_cutoff = 4.8
correlation_cutoff = 0.8
q20_cutoff = 90
q30_cutoff = 95
gc_cutoff = [40,55]
mapped_reads_cutoff = 90
unique_mapped_reads_cutoff = 80
multi_mappedreads_cutoff = 20
exon_cut_off = 80
intron_cut_off= 10
intergenic_cut_off= 10

# 数据格式转换
samples = allDataOut['Sample']
Read1WithAdapter = allDataOut['Read1WithAdapter'].str.strip('%').astype('float')
Read2WithAdapter = allDataOut['Read2WithAdapter'].str.strip('%').astype('float')
PairsThatWereTooShort = allDataOut['PairsThatWereTooShort'].str.strip('%').astype('float')
RawData = allDataOut['RawData'].str.strip('G').astype('float')
CleanData = allDataOut['CleanData'].str.strip('G').astype('float')

# 基本指标判定
read1WithAdapter_error = len(samples[Read1WithAdapter > read1_with_adapter_cutoff])
read2WithAdapter_error = len(samples[Read2WithAdapter > read2_with_adapter_cutoff])
pairsThatWereTooShort_error = len(samples[PairsThatWereTooShort > pairs_that_were_too_short_cutoff])
rawData_error = len(samples[RawData <= raw_data_cutoff])
cleanData_error = len(samples[CleanData <= clean_data_cutoff])
q20_error = len(samples[allDataOut['Q20']< q20_cutoff])
q30_error = len(samples[allDataOut['Q30']< q30_cutoff])
gc_error = len(samples[(allDataOut['GC'] <= gc_cutoff[0]) | (allDataOut['GC'] >= gc_cutoff[1])])
mapped_reads_error = len(samples[allDataOut['Mappedreads'].astype('float') < mapped_reads_cutoff])
unique_mapped_error = len(samples[allDataOut['UniqueMappedreads'].astype('float') < unique_mapped_reads_cutoff])
multi_mappedreads_error = len(samples[allDataOut['MultiMappedreads'].astype('float') > multi_mappedreads_cutoff])
exon_error = len(samples[allDataOut['exon'].astype('float') <= exon_cut_off])
intron_error = len(samples[allDataOut['intron'].astype('float') >= intron_cut_off])
intergenic_error = len(samples[allDataOut['intergenic'].astype('float') >= intergenic_cut_off])

# 相关性判定
cor_cond = pd.DataFrame(list(allDataOut['CorrelationOfSample'].str.split(';')), dtype=float)

# 某些组的样本可能有1个或者2个总之存在不同组样本数量不一致的现象，如果放置在同一个数据框中会出现NaN的情况，因此进行数据填充
cor_cond.fillna(1, inplace=True)

# 使用any(1)结合布尔索引按行判断
cor_filter = cor_cond[(cor_cond <= correlation_cutoff).any(1)]
cor_error = len(cor_filter)

# 数据合并
data_check = pd.Series({
'SampleNum': len(allDataOut),
'RawData': rawData_error,
'CleanData': cleanData_error,
'Q20': q20_error,
'Q30': q30_error,
'GC': gc_error,
'MappedReads': mapped_reads_error,
'UniqueMappedReads': unique_mapped_error,
'Read1WithAdapter': read1WithAdapter_error,
'Read2WithAdapter': read2WithAdapter_error,
'PairsThatWereTooShort': pairsThatWereTooShort_error,
'CorrelationOfSamples': cor_error,
'MultiMappedReads': multi_mappedreads_error,
'Exon': exon_error,
'Intron': intron_error,
'intergenic': intergenic_error})

# 统计异常比例
error_ratio = data_check.values.astype('int')/len(allDataOut)

# 形成数据框
error_stat = pd.DataFrame([error_ratio], columns=list(data_check.index))
f = lambda x: '{0:.2f}%'.format(x*100)
error_stat_out = pd.DataFrame([error_stat.iloc[0,:].map(f)])
error_stat_out[[ 'SampleNum','合同号', '路径',]] = [len(allDataOut),itemNumber,itemPath]
check_output = '/public23/4_After_Sales_Personal/YangPei/20.Script/Python_script/refRNA_stat/test/{0}_check_data.txt'.format(itemNumber)
error_stat_out.to_csv(check_output, sep='\t', index=False)

