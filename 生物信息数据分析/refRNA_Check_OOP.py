import os
import re
import sys
import argparse
import subprocess
import pandas as pd

"""
命令行参数1: 工作路径
命令行参数2: 产品类型

"""
# ========================== 创建命令行参数 =========================
# 初始化argparse对象
parser = argparse.ArgumentParser(
                    prog='DataCheck',
                    description='This Program Is Checking Data Quality',
                    epilog='Text at the bottom of help')
# 创建实例对象
parser.add_argument('workDir', help='please provide a work path')  # 工作路径
parser.add_argument('itemType', help='please provide this project type')  # 工作路径
parser.add_argument('--outDir', default=os.getcwd(), help='please provide a path for output')  # 结果文件输出路径

# 解析参数
args = parser.parse_args()

# 导出参数
itemPath = args.workDir
itemType = args.itemType

class RefRNA_Check:
    def __init__(self, itemPath):
        self.itemPath = itemPath
        self.sample_info =  pd.read_csv(r'{0}/sample_info.txt'.format(self.itemPath),sep='\t').drop_duplicates()
        self.sample_info.rename(columns = {'#SampleID':'Sample','COND1':'COND1'}, inplace=True)

    def path_exists(self):
        sample = self.sample_info.iloc[0, 0]
        self.sample_path = os.path.exists(r'{0}/sample_info.txt'.format(self.itemPath))
        self.project_info_path = os.path.exists(r'{0}/project_info/04_report.txt'.format(self.itemPath))
        self.clean_data_path = os.path.exists('{0}/{1}/{2}/{2}_delete_adapter.summary'.format(self.itemPath,'CleanData', sample))
        self.sampleCor_path = os.path.exists('{0}/Output/merged_result/correlation_cluster.txt'.format(self.itemPath))
        self.dataStat_path = os.path.exists('{0}/Output/stat_out.txt'.format(self.itemPath))
        self.mappedStat_path = os.path.exists('{0}/Output/mapped_stat_out.txt'.format(self.itemPath))
        self.mappedRegion_path = '{0}/Output/mapped_region_stat.txt'.format(self.itemPath)
        self.strandStat_path = os.path.exists('Output/{0}/RSeQC_result/{0}_Strand_specific.log'.format(sample))
        self.marcb_path = os.path.exists('CleanData/{0}/{0}_bowtie_abundance_1.log'.format(sample))

    def read_library(self):
        samples = pd.read_csv(r'{0}/project_info/01_samples.txt'.format(self.itemPath),sep='\t').drop_duplicates()
        projectInfo = samples.iloc[:, [1, 2]]
        projectInfo.columns = ['文库名称', 'Sample']
        self.libInfo = pd.merge(self.sample_info, projectInfo, on='Sample')
        return self.libInfo
                
    def read_short_read(self, sample):
        dics = {}
        patternOne = re.compile(r'  Read 1 with adapter:\s+.+\s+(.+%)')
        patternTwo = re.compile(r'  Read 2 with adapter:\s+.+\s+(.+%)')
        patternThree = re.compile(r'Pairs that were too short:\s+.+\s+(.+%)')
        cleanDataPath = r'{0}/{1}/{2}/{3}_delete_adapter.summary'.format(self.itemPath,'CleanData', sample, sample)
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
        self.df_merge = pd.DataFrame(dics)
        return self.df_merge
    
    def sampleCor(self, group_name):
        corData = pd.read_csv(r'{0}/Output/merged_result/correlation_cluster.txt'.format(self.itemPath), sep='\t', index_col=0)
        group_name_bool = self.sample_info.iloc[:,1] == group_name # 筛选组名所在的行，返回布尔值
        sample_index = self.sample_info.iloc[:, 0][group_name_bool] # 根据组名返回的布尔值，筛选与组名对应的样本名
        rowData = corData.loc[list(sample_index.values), list(sample_index.values)]

        # 按照样本筛选每个样本与其他样本的相关性
        sampleData = rowData.columns.map(lambda x: ';'.join(rowData.loc[:, x].values.astype(str)))
        self.sampleCorInfoResult = pd.DataFrame(sampleData, index=rowData.columns, columns=['CorrelationOfSample'])
        return self.sampleCorInfoResult 

    def stat_out(self):
        statData = pd.read_csv(r'{0}/Output/stat_out.txt'.format(self.itemPath), sep='\t', header=0)
        self.newstatData = statData.iloc[1:, [0, 2, 4, 5, 6, 7, 8]]
        self.newstatData.columns = ['Sample', 'RawData', 'CleanData', 'ValidRatio', 'Q20', 'Q30', 'GC']
        return self.newstatData
    
    def mapped_stat(self):
        statData = pd.read_csv('{0}/Output/mapped_stat_out.txt'.format(self.itemPath), sep='\t', header=0)
        selectData = statData.loc[:, ['Sample', 'Mapped reads', 'Unique Mapped reads', 'Multi Mapped reads']]
        selectData['Mappedreads'] = selectData['Mapped reads'].str.extract(r'([0-9]+.[0-9]+)%')
        selectData['UniqueMappedreads'] = selectData['Unique Mapped reads'].str.extract(r'([0-9]+.[0-9]+)%')
        selectData['MultiMappedreads'] = selectData['Multi Mapped reads'].str.extract(r'([0-9]+.[0-9]+)%')
        self.mapped_stat_result = selectData.loc[:, ['Sample', 'Mappedreads', 'UniqueMappedreads', 'MultiMappedreads']]
        return self.mapped_stat_result

    def mapped_region(self):
        regionData = pd.read_csv('{0}/Output/mapped_region_stat.txt'.format(self.itemPath), sep='\t', header=0, index_col=0)
        self.regionDataTranspose = regionData.transpose()
        self.regionDataTranspose['Sample'] = self.regionDataTranspose.index
        return self.regionDataTranspose


    def strand_info(self):
        try:
            cmd1 = "cut -f1 sample_info.txt | grep -v '#'"
            stdout = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, encoding='utf-8')
            strand_data = []
            for sample in stdout.stdout.readlines():
                sample = sample.strip('\n')
                cmd2 = "grep '1+-,1-+,2++,2--' Output/{0}/RSeQC_result/{0}_Strand_specific.log".format(sample)
                strand_cmd_out = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, encoding='utf-8')
                strand_specific_info = strand_cmd_out.stdout.readlines()
                strand_specific_ratio = strand_specific_info[0][-7:-1]
                result = [sample, strand_specific_ratio]
                strand_data.append(result)
            self.strand_out = pd.DataFrame(strand_data,columns=['Sample', 'StrandInfo'])
            return self.strand_out
        except IndexError:
            self.strand_out = pd.DataFrame({'Sample': self.sample_info.iloc[:,0],'链特异性': None})
            return self.strand_out

    def marcb_info(self):
        try:
            pattern = '.+/(.+)/.+:([0-9]+.[0-9]+%) .+'
            cmd = 'grep overall CleanData/*/*_bowtie_abundance_1.log'
            marcb_cmd_out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, encoding='utf-8')
            df_list = []
            for line in marcb_cmd_out.stdout.readlines():
                result = re.findall(pattern, line)
                df = pd.DataFrame({'Sample':result[0][0], 'marcbRatio': [result[0][1]]})
                df_list.append(df)
            self.marcb_out = pd.concat(df_list, axis=0)
            return self.marcb_out
        except ValueError:
            self.marcb_out = pd.DataFrame({'Sample': self.sample_info.iloc[:,0],'marcbRatio': None})
            return self.marcb_out
    
    def item_info(self):
        reportData = pd.read_csv('{0}/project_info/04_report.txt'.format(self.itemPath), sep='\t', header=None)
        customer_name = reportData.iloc[0, 1]
        self.itemNum = reportData.iloc[3, 1]
        self.itemInfo = pd.DataFrame({'Sample': self.sample_info.iloc[:,0], '姓名': customer_name, '项目编号': self.itemNum, '项目路径': self.itemPath})
        return self.itemInfo, self.itemNum



if __name__ == '__main__':
    refRNA_check = RefRNA_Check(itemPath='/Users/yangpei/YangPei/after_sale/test/refRNA')
    refRNA_check.path_exists()
    """输出文库信息"""
    empty_libInfo = pd.DataFrame({'Sample': refRNA_check.sample_info.iloc[:,0], '文库名称': None})
    if refRNA_check.project_info_path:
        libStat = refRNA_check.read_library()
        if not libStat.empty:
            libInfo = libStat.iloc[:,[0,2]]
        else:
            libInfo = empty_libInfo
    else:
        libInfo = empty_libInfo

    """短片段比例"""
    empty_cleanDataStatInfo = pd.DataFrame({'Sample':  refRNA_check.sample_info.iloc[:,0], 'Read1WithAdapter': None,
                                 'Read2WithAdapter': None, 'PairsThatWereTooShort': None})
    if refRNA_check.clean_data_path:
        samples = refRNA_check.sample_info.iloc[:,0]
        cleanDataInfoResult = samples.map(refRNA_check.read_short_read)
        cleanDataStat = pd.concat(cleanDataInfoResult.values)
        if not cleanDataStat.empty:
            cleanDataStatInfo = cleanDataStat
        else:
            cleanDataStatInfo = empty_cleanDataStatInfo
    else:
        cleanDataStatInfo = empty_cleanDataStatInfo
    
    """样本相关性"""
    empty_CorrelationInfo = pd.DataFrame({'Sample': refRNA_check.sample_info.iloc[:,0], 'CorrelationOfSample': None})
    if refRNA_check.sampleCor_path:
        groups = refRNA_check.sample_info.iloc[:,1].unique()
        sampleCorResult = pd.Series(groups).map(refRNA_check.sampleCor)
        CorrelationData = pd.concat(sampleCorResult.values)
        CorrelationData['Sample'] = CorrelationData.index
        if not CorrelationData.empty:
            CorrelationInfo = CorrelationData
        else:
            CorrelationInfo = empty_CorrelationInfo
    else:
        CorrelationInfo = empty_CorrelationInfo

    """样本数据量统计"""
    empty_dataStatInfo = pd.DataFrame({'Sample': refRNA_check.sample_info.iloc[:,0], 'RawData': None, 'CleanData': None,
                                    'Q20': None, 'Q30': None, 'GC': None})
    if refRNA_check.dataStat_path:
        dataStat = refRNA_check.stat_out()
        if not dataStat.empty:
            dataStatInfo = dataStat
        else:
            dataStatInfo = empty_dataStatInfo
    else:
        dataStatInfo = empty_dataStatInfo

    """基因组比对统计"""
    empty_mappedStatInfo = pd.DataFrame({'Sample': refRNA_check.sample_info.iloc[:,0], 'Mappedreads': None, 
                                      'UniqueMappedreads': None, 'MultiMappedreads': None})
    if refRNA_check.mappedStat_path:
        mappedStat = refRNA_check.mapped_stat()
        if not mappedStat.empty:
            mappedStatInfo = mappedStat
        else:
            mappedStatInfo = empty_mappedStatInfo
    else:
        mappedStatInfo = empty_mappedStatInfo
    
    """基因区间比对区域统计"""
    empty_mappedRegionInfo = pd.DataFrame({'Sample': refRNA_check.sample_info.iloc[:,0], 'exon': None, 'intron': None, 'intergenic': None})
    if refRNA_check.mappedRegion_path:
        mappedRegionStat = refRNA_check.mapped_region()
        if not mappedRegionStat.empty:
            mappedRegionInfo = mappedRegionStat
        else:
            mappedRegionInfo = empty_mappedRegionInfo
    else:
        mappedRegionInfo = empty_mappedRegionInfo


    """链特异性"""
    empty_strandStatInfo = pd.DataFrame({'Sample': refRNA_check.sample_info.iloc[:,0],'链特异性': None})
    if refRNA_check.strandStat_path:
        strandStat = refRNA_check.strand_info()
        if not strandStat.empty:
            strandStatInfo = strandStat
        else:
            strandStatInfo = empty_strandStatInfo
    else:
        strandStatInfo = empty_strandStatInfo

    """核糖体占比"""
    empty_marcbStatInfo = pd.DataFrame({'Sample': refRNA_check.sample_info.iloc[:,0],'marcbRatio': None})
    if refRNA_check.marcb_path:
        marcbStat = refRNA_check.marcb_info()
        if not marcbStat.empty:
            marcbStatInfo = marcbStat
        else:
            marcbStatInfo = empty_marcbStatInfo
    else:
        marcbStatInfo = empty_marcbStatInfo

    """项目信息"""
    empty_itemStatInfo = pd.DataFrame({'Sample': refRNA_check.sample_info.iloc[:,0], '项目编号': None, '项目路径': None})
    
    if refRNA_check.project_info_path:
        itemStat = refRNA_check.item_info()
        if not itemStat[0].empty:
            itemInfo = itemStat[0]
        else:
            itemStatInfo = empty_itemStatInfo
    else:
        itemStatInfo = empty_itemStatInfo


    """文件输出"""
    allDataOut = pd.DataFrame({'Sample': refRNA_check.sample_info.iloc[:,0]})

    for tmp in [libInfo, cleanDataStatInfo, CorrelationInfo, dataStatInfo, mappedStatInfo,
                mappedRegionInfo, strandStatInfo, marcbStatInfo, itemInfo]:
        allDataOut = pd.merge(allDataOut, tmp, on='Sample')



if itemType == 'refRNA' or itemType == 'lncRNA':
    os.chdir(itemPath)
    RefRNA_Check = RefRNA_Check(itemPath=itemPath)
    sample = RefRNA_Check.sample_info.iloc[0,0]

    """输出文库信息"""
    libInfo = RefRNA_Check.read_library()
    if os.path.exists(r'{0}/sample_info.txt'.format(itemPath)) and not libInfo.empty:
        libInfo = libInfo.iloc[:,[0,2]]    
    else:
        libInfo = pd.DataFrame({'Sample': RefRNA_Check.sample_info.iloc[:,0], '文库名称': None})
        
    """客户姓名"""
    customerName = RefRNA_Check.read_customer_name()
    if os.path.exists(r'{0}/project_info/04_report.txt'.format(itemPath)) and not customerName.empty:
        customerInfo = customerName 
    else:
        customerInfo = pd.DataFrame({'Sample': RefRNA_Check.sample_info.iloc[:,0], '姓名': None})

    """短片段比例"""
    samples = RefRNA_Check.sample_info.iloc[:,0]
    cleanDataInfoResult = samples.map(RefRNA_Check.read_short_read)
    cleanDataStat = pd.concat(cleanDataInfoResult.values)

    if os.path.exists('{0}/{1}/{2}/{2}_delete_adapter.summary'.format(itemPath,'CleanData', sample)) and not cleanDataStat.empty:
        cleanDataStatInfo = cleanDataStat 
    else:
        cleanDataStatInfo = pd.DataFrame({'Sample':  RefRNA_Check.sample_info.iloc[:,0], 'Read1WithAdapter': None,
                                 'Read2WithAdapter': None, 'PairsThatWereTooShort': None})
    
    """样本相关性"""
    groups = RefRNA_Check.sample_info.iloc[:,1].unique()
    sampleCorResult = pd.Series(groups).map(RefRNA_Check.sampleCor)
    CorrelationData = pd.concat(sampleCorResult.values)
    CorrelationData['Sample'] = CorrelationData.index

    if os.path.exists('{0}/Output/merged_result/correlation_cluster.txt'.format(itemPath)) and not CorrelationData.empty:
        CorrelationInfo = CorrelationData 
    else:
        CorrelationInfo = pd.DataFrame({'Sample': RefRNA_Check.sample_info.iloc[:,0], 'CorrelationOfSample': None})

    """样本数据量统计"""
    dataStat = RefRNA_Check.stat_out()
    if os.path.exists('{0}/Output/stat_out.txt'.format(itemPath)) and not dataStat.empty:
        dataStatInfo = dataStat 
    else:
        dataStatInfo = pd.DataFrame({'Sample': RefRNA_Check.sample_info.iloc[:,0], 'RawData': None, 'CleanData': None,
                                    'Q20': None, 'Q30': None, 'GC': None})

    """基因组比对统计"""
    mappedStat = RefRNA_Check.mapped_stat()
    if os.path.exists('{0}/Output/mapped_stat_out.txt'.format(itemPath)) and not mappedStat.empty:
        mappedStatInfo = mappedStat 
    else:
        mappedStatInfo = pd.DataFrame({'Sample': RefRNA_Check.sample_info.iloc[:,0], 'Mappedreads': None, 
                                      'UniqueMappedreads': None, 'MultiMappedreads': None})
    
    """基因组比对区域统计"""
    mappedRegionStat = RefRNA_Check.mapped_region()
    if os.path.exists('{0}/Output/mapped_region_stat.txt'.format(itemPath)) and not mappedRegionStat.empty:
        mappedRegionInfo = mappedRegionStat 
    else:
        mappedRegionInfo = pd.DataFrame({'Sample': RefRNA_Check.sample_info.iloc[:,0], 'exon': None, 'intron': None, 'intergenic': None})

    """链特异性"""
    strandStat = RefRNA_Check.strand_info()
    if os.path.exists('Output/{0}/RSeQC_result/{0}_Strand_specific.log'.format(sample)) and not strandStat.empty:
        strandStatInfo = strandStat 
    else:
        strandStatInfo = pd.DataFrame({'Sample': RefRNA_Check.sample_info.iloc[:,0],'链特异性': None})

    """核糖体占比"""
    marcbStat = RefRNA_Check.marcb_info()
    if os.path.exists('CleanData/{0}/{0}_bowtie_abundance_1.log'.format(sample)) and not marcbStat.empty:
        marcbStatInfo = marcbStat 
    else:
        marcbStatInfo = pd.DataFrame({'Sample': RefRNA_Check.sample_info.iloc[:,0],'marcbRatio': None})

    """项目信息"""
    itemStat = RefRNA_Check.item_info()

    if os.path.exists('{0}/project_info/04_report.txt'.format(itemPath)) and not itemStat:
        itemStatInfo = itemStat[0]
    else:
        itemStatInfo = pd.DataFrame({'Sample': RefRNA_Check.sample_info.iloc[:,0], '项目编号': None, '项目路径': None})


    """文件输出"""
    pattern = r'(LC-P|USA-)[0-9]+(-[0-9]){0,1}(_LC-P[0-9]+){0,1}_[0-9]+'
    itemNumbers = re.search(pattern, RefRNA_Check.itemPath).group()
    output_name = '{0}/{1}_data_stat.txt'.format(args.outDir, itemNumbers)
    xlsx_name = '{0}/{1}_data_stat.xlsx'.format(args.outDir, itemNumbers)


    allDataOut = pd.DataFrame({'Sample': RefRNA_Check.sample_info.iloc[:,0]})

    for tmp in [libInfo, cleanDataStatInfo, CorrelationInfo, dataStatInfo, mappedStatInfo,
                mappedRegionInfo, strandStatInfo, marcbStatInfo, customerInfo, itemStatInfo]:
        allDataOut = pd.merge(allDataOut, tmp, on='Sample')
    allDataOut.to_csv(output_name, sep='\t', index=False)
    allDataOut.to_excel(xlsx_name, index=False, engine='openpyxl')



# lncRNA_CircRNA
elif itemType == 'lncRNA_circRNA':
    class LncRNA_Check(RefRNA_Check):
        def __init__(self, itemPath):
            RefRNA_Check.__init__(self, itemPath + '/lncRNA')
        
        def read_library(self):
            samples = pd.read_csv(r'project_info/01_samples.txt',sep='\t').drop_duplicates()
            projectInfo = samples.iloc[:, [1, 2]]
            projectInfo.columns = ['文库名称', 'Sample']
            self.libInfo = pd.merge(self.sample_info, projectInfo, on='Sample')
            return self.libInfo
            
        def read_customer_name(self):
            customer_info = pd.read_csv(r'project_info/04_report.txt',sep='\t',header=None)
            customer_name = customer_info.iloc[0,1]
            self.nameInfo = pd.DataFrame({'Sample': self.sample_info.iloc[:,0], '姓名': customer_name})
            return self.nameInfo
            
        def item_info(self):
            reportData = pd.read_csv('project_info/04_report.txt', sep='\t', header=None, index_col=0)
            self.itemNum = reportData.loc['LCB_项目编号', 1]
            self.itemInfo = pd.DataFrame({'Sample': self.sample_info.iloc[:,0], '项目编号': self.itemNum, '项目路径': self.itemPath})
            return self.itemInfo, self.itemNum
    
    os.chdir(itemPath)
    LncRNA_Check = LncRNA_Check(itemPath=itemPath)
    sample = LncRNA_Check.sample_info.iloc[0,0]

    """输出文库信息"""
    libInfo = LncRNA_Check.read_library()
    if os.path.exists(r'{0}/sample_info.txt'.format(itemPath)) and not libInfo.empty:
        libInfo = libInfo.iloc[:,[0,2]]    
    else:
        libInfo = pd.DataFrame({'Sample': LncRNA_Check.sample_info.iloc[:,0], '文库名称': None})
        
    """客户姓名"""
    customerName = LncRNA_Check.read_customer_name()
    if os.path.exists(r'{0}/project_info/04_report.txt'.format(itemPath)) and not customerName.empty:
        customerInfo = customerName 
    else:
        customerInfo = pd.DataFrame({'Sample': LncRNA_Check.sample_info.iloc[:,0], '姓名': None})

    """短片段比例"""
    samples = LncRNA_Check.sample_info.iloc[:,0]
    cleanDataInfoResult = samples.map(LncRNA_Check.read_short_read)
    cleanDataStat = pd.concat(cleanDataInfoResult.values)

    if os.path.exists('{0}/{1}/{2}/{2}_delete_adapter.summary'.format(itemPath,'CleanData', sample)) and not cleanDataStat.empty:
        cleanDataStatInfo = cleanDataStat 
    else:
        cleanDataStatInfo = pd.DataFrame({'Sample':  LncRNA_Check.sample_info.iloc[:,0], 'Read1WithAdapter': None,
                                 'Read2WithAdapter': None, 'PairsThatWereTooShort': None})
    
    """样本相关性"""
    groups = LncRNA_Check.sample_info.iloc[:,1].unique()
    sampleCorResult = pd.Series(groups).map(LncRNA_Check.sampleCor)
    CorrelationData = pd.concat(sampleCorResult.values)
    CorrelationData['Sample'] = CorrelationData.index

    if os.path.exists('{0}/Output/merged_result/correlation_cluster.txt'.format(itemPath)) and not CorrelationData.empty:
        CorrelationInfo = CorrelationData 
    else:
        CorrelationInfo = pd.DataFrame({'Sample': LncRNA_Check.sample_info.iloc[:,0], 'CorrelationOfSample': None})

    """样本数据量统计"""
    dataStat = LncRNA_Check.stat_out()
    if os.path.exists('{0}/Output/stat_out.txt'.format(itemPath)) and not dataStat.empty:
        dataStatInfo = dataStat 
    else:
        dataStatInfo = pd.DataFrame({'Sample': LncRNA_Check.sample_info.iloc[:,0], 'RawData': None, 'CleanData': None,
                                    'Q20': None, 'Q30': None, 'GC': None})

    """基因组比对统计"""
    mappedStat = LncRNA_Check.mapped_stat()
    if os.path.exists('{0}/Output/mapped_stat_out.txt'.format(itemPath)) and not mappedStat.empty:
        mappedStatInfo = mappedStat 
    else:
        mappedStatInfo = pd.DataFrame({'Sample': LncRNA_Check.sample_info.iloc[:,0], 'Mappedreads': None, 
                                      'UniqueMappedreads': None, 'MultiMappedreads': None})
    
    """基因组比对区域统计"""
    mappedRegionStat = LncRNA_Check.mapped_region()
    if os.path.exists('{0}/Output/mapped_region_stat.txt'.format(itemPath)) and not mappedRegionStat.empty:
        mappedRegionInfo = mappedRegionStat 
    else:
        mappedRegionInfo = pd.DataFrame({'Sample': LncRNA_Check.sample_info.iloc[:,0], 'exon': None, 'intron': None, 'intergenic': None})

    """链特异性"""
    strandStat = LncRNA_Check.strand_info()
    if os.path.exists('Output/{0}/RSeQC_result/{0}_Strand_specific.log'.format(sample)) and not strandStat.empty:
        strandStatInfo = strandStat 
    else:
        strandStatInfo = pd.DataFrame({'Sample': LncRNA_Check.sample_info.iloc[:,0],'链特异性': None})

    """核糖体占比"""
    marcbStat = LncRNA_Check.marcb_info()
    if os.path.exists('CleanData/{0}/{0}_bowtie_abundance_1.log'.format(sample)) and not marcbStat.empty:
        marcbStatInfo = marcbStat 
    else:
        marcbStatInfo = pd.DataFrame({'Sample': LncRNA_Check.sample_info.iloc[:,0],'marcbRatio': None})

    """项目信息"""
    itemStat = LncRNA_Check.item_info()

    if os.path.exists('{0}/project_info/04_report.txt'.format(itemPath)) and not itemStat:
        itemStatInfo = itemStat[0]
    else:
        itemStatInfo = pd.DataFrame({'Sample': LncRNA_Check.sample_info.iloc[:,0], '项目编号': None, '项目路径': None})


    """文件输出"""
    pattern = r'(LC-P|USA-)[0-9]+(-[0-9]){0,1}(_LC-P[0-9]+){0,1}_[0-9]+'
    itemNumbers = re.search(pattern, LncRNA_Check.itemPath).group()
    output_name = '{0}/{1}_data_stat.txt'.format(args.outDir, itemNumbers)
    xlsx_name = '{0}/{1}_data_stat.xlsx'.format(args.outDir, itemNumbers)


    allDataOut = pd.DataFrame({'Sample': LncRNA_Check.sample_info.iloc[:,0]})

    for tmp in [libInfo, cleanDataStatInfo, CorrelationInfo, dataStatInfo, mappedStatInfo,
                mappedRegionInfo, strandStatInfo, marcbStatInfo, customerInfo, itemStatInfo]:
        allDataOut = pd.merge(allDataOut, tmp, on='Sample')
    allDataOut.to_csv(output_name, sep='\t', index=False)
    allDataOut.to_excel(xlsx_name, index=False, engine='openpyxl')
    
#circRNA
elif itemType == 'circRNA':
    class CircRNA_Check(RefRNA_Check):
        pass

# 路径判断

sample_path = os.path.exists(r'{0}/sample_info.txt'.format(itemPath))
project_info_path = os.path.exists(r'{0}/project_info/04_report.txt'.format(itemPath))
clean_data_path = os.path.exists('{0}/{1}/{2}/{2}_delete_adapter.summary'.format(itemPath,'CleanData', sample))
sampleCor_path = os.path.exists('{0}/Output/merged_result/correlation_cluster.txt'.format(itemPath))
dataStat_path = os.path.exists('{0}/Output/stat_out.txt'.format(itemPath))
mappedStat_path = os.path.exists('{0}/Output/mapped_stat_out.txt'.format(itemPath))
mappedRegion_path = '{0}/Output/mapped_region_stat.txt'.format(itemPath)
strandStat_path = os.path.exists('Output/{0}/RSeQC_result/{0}_Strand_specific.log'.format(sample))
marcb_path = os.path.exists('CleanData/{0}/{0}_bowtie_abundance_1.log'.format(sample))
