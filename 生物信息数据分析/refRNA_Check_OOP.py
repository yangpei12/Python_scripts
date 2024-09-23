import os
import re
import sys
import subprocess
import pandas as pd

"""
命令行参数1: 工作路径
命令行参数2: 产品类型
"""
argvs = sys.argv
itemPath = '/Users/yangpei/YangPei/after_sale/test/refRNA'
#projectType = argvs[2]

"""
构造函数创建sample_info的信息、路径信息
函数1：文库信息
函数2：客户姓名信息
函数3：短片段和接头比例
函数4：相关性
函数5：读取stat_out
函数6：读取mapped_stat数据
函数7：读取mapped_region数据
函数8：读取链条特异性信息
函数9：读取核糖体占比信息
函数10：读取项目号及路径
"""

class Ref_Rna_Check:
    def __init__(self, itemPath):
        self.itemPath = itemPath
        self.sample_info =  pd.read_csv(r'{0}/sample_info.txt'.format(self.itemPath),sep='\t').drop_duplicates()
        self.sample_info.rename(columns = {'#SampleID':'Sample','COND1':'COND1'}, inplace=True)

    
    def read_library(self):
        samples = pd.read_csv(r'{0}/project_info/01_samples.txt'.format(self.itemPath),sep='\t').drop_duplicates()
        projectInfo = samples.iloc[:, [1, 2]]
        projectInfo.columns = ['文库名称', 'Sample']
        self.libInfo = pd.merge(self.sample_info, projectInfo, on='Sample')
        return self.libInfo
        
    def read_customer_name(self):
        customer_info = pd.read_csv(r'{0}/project_info/04_report.txt'.format(self.itemPath),sep='\t',header=None)
        customer_name = customer_info.iloc[0,1]
        self.nameInfo = pd.DataFrame({'Sample': self.sample_info.iloc[:,0], '姓名': customer_name})
        return self.nameInfo
        
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
        reportData = pd.read_csv('{0}/project_info/04_report.txt'.format(self.itemPath), sep='\t', header=None, index_col=0)
        self.itemNum = reportData.loc['LCB_项目编号', 1]
        self.itemInfo = pd.DataFrame({'Sample': self.sample_info.iloc[:,0], '项目编号': self.itemNum, '项目路径': self.itemPath})
        return self.itemInfo, self.itemNum



if __name__ == '__main__':
    ref_rna_check = Ref_Rna_Check(itemPath=itemPath)
    sample = ref_rna_check.sample_info.iloc[0,0]

    """输出文库信息"""
    libInfo = ref_rna_check.read_library()
    if os.path.exists(r'{0}/sample_info.txt'.format(itemPath)) and not libInfo.empty:
        libInfo = libInfo.iloc[:,[0,2]]    
    else:
        libInfo = pd.DataFrame({'Sample': ref_rna_check.sample_info.iloc[:,0], '文库名称': None})
        
    """客户姓名"""
    customerName = ref_rna_check.read_customer_name()
    if os.path.exists(r'{0}/project_info/04_report.txt'.format(itemPath)) and not customerName.empty:
        customerInfo = customerName 
    else:
        customerInfo = pd.DataFrame({'Sample': ref_rna_check.sample_info.iloc[:,0], '姓名': None})

    """短片段比例"""
    samples = ref_rna_check.sample_info.iloc[:,0]
    cleanDataInfoResult = samples.map(ref_rna_check.read_short_read)
    cleanDataStat = pd.concat(cleanDataInfoResult.values)

    if os.path.exists('{0}/{1}/{2}/{2}_delete_adapter.summary'.format(itemPath,'CleanData', sample)) and not cleanDataStat.empty:
        cleanDataStatInfo = cleanDataStat 
    else:
        cleanDataStatInfo = pd.DataFrame({'Sample':  ref_rna_check.sample_info.iloc[:,0], 'Read1WithAdapter': None,
                                 'Read2WithAdapter': None, 'PairsThatWereTooShort': None})
    
    """样本相关性"""
    groups = ref_rna_check.sample_info.iloc[:,1].unique()
    sampleCorResult = pd.Series(groups).map(ref_rna_check.sampleCor)
    CorrelationData = pd.concat(sampleCorResult.values)
    CorrelationData['Sample'] = CorrelationData.index

    if os.path.exists('{0}/Output/merged_result/correlation_cluster.txt'.format(itemPath)) and not CorrelationData.empty:
        CorrelationInfo = CorrelationData 
    else:
        CorrelationInfo = pd.DataFrame({'Sample': ref_rna_check.sample_info.iloc[:,0], 'CorrelationOfSample': None})

    """样本数据量统计"""
    dataStat = ref_rna_check.stat_out()
    if os.path.exists('{0}/Output/stat_out.txt'.format(itemPath)) and not dataStat.empty:
        dataStatInfo = dataStat 
    else:
        dataStatInfo = pd.DataFrame({'Sample': ref_rna_check.sample_info.iloc[:,0], 'RawData': None, 'CleanData': None,
                                    'Q20': None, 'Q30': None, 'GC': None})

    """基因组比对统计"""
    mappedStat = ref_rna_check.mapped_stat()
    if os.path.exists('{0}/Output/mapped_stat_out.txt'.format(itemPath)) and not mappedStat.empty:
        mappedStatInfo = mappedStat 
    else:
        mappedStatInfo = pd.DataFrame({'Sample': ref_rna_check.sample_info.iloc[:,0], 'Mappedreads': None, 
                                      'UniqueMappedreads': None, 'MultiMappedreads': None})
    
    """基因组比对区域统计"""
    mappedRegionStat = ref_rna_check.mapped_region()
    if os.path.exists('{0}/Output/mapped_region_stat.txt'.format(itemPath)) and not mappedRegionStat.empty:
        mappedRegionInfo = mappedRegionStat 
    else:
        mappedRegionInfo = pd.DataFrame({'Sample': ref_rna_check.sample_info.iloc[:,0], 'exon': None, 'intron': None, 'intergenic': None})

    """链特异性"""
    strandStat = ref_rna_check.strand_info()
    if os.path.exists('Output/{0}/RSeQC_result/{0}_Strand_specific.log'.format(sample)) and not strandStat.empty:
        strandStatInfo = strandStat 
    else:
        strandStatInfo = pd.DataFrame({'Sample': ref_rna_check.sample_info.iloc[:,0],'链特异性': None})

    """核糖体占比"""
    marcbStat = ref_rna_check.marcb_info()
    if os.path.exists('CleanData/{0}/{0}_bowtie_abundance_1.log'.format(sample)) and not marcbStat.empty:
        marcbStatInfo = marcbStat 
    else:
        marcbStatInfo = pd.DataFrame({'Sample': ref_rna_check.sample_info.iloc[:,0],'marcbRatio': None})

    """项目信息"""
    itemStat = ref_rna_check.item_info()

    if os.path.exists('Output/merged_result/correlation_cluster.txt') and not itemStat:
        itemInfo = itemStat[0]
    else:
        itemStatInfo = pd.DataFrame({'Sample': ref_rna_check.sample_info.iloc[:,0], '项目编号': None, '项目路径': None})


    """文件输出"""
    pattern = r'(LC-P|USA-)[0-9]+(-[0-9]){0,1}(_LC-P[0-9]+){0,1}_[0-9]+'
    itemNumbers = re.search(pattern, ref_rna_check.itemPath).group()
    output_name = '{0}/{1}_data_stat.txt'.format(args.outDir, itemNumbers)
    xlsx_name = '{0}/{1}_data_stat.xlsx'.format(args.outDir, itemNumbers)


    allDataOut = pd.DataFrame({'Sample': ref_rna_check.sample_info.iloc[:,0]})

    for tmp in [libInfo, cleanDataStatInfo, CorrelationInfo, dataStatInfo, mappedStatInfo,
                mappedRegionInfo, strandStatInfo, marcbStatInfo, customerInfo, itemInfo]:
        allDataOut = pd.merge(allDataOut, tmp, on='Sample')
    allDataOut.to_csv(output_name, sep='\t', index=False)
    allDataOut.to_excel(xlsx_name, index=False, engine='openpyxl')

