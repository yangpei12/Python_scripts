import os
import pandas as pd
import subprocess

# 工作路径
os.chdir(r'/mnt/d/售后/蓝芳仁/ata分析')

# 文件准备BED、wig、gga_refseq_IDmapping.txt
work_path = r'/mnt/d/售后/蓝芳仁/ata分析'
wig_file_path = '{0}/wig_file/gga.bed'.format(work_path)
stat_out_path = '{0}/stat_out.txt'.format(work_path)

# 生成mapping_wig_location_with_depth.txt
stat_out_data = pd.read_csv(stat_out_path, sep='\t')
sample_data = stat_out_data.iloc[1:,[0, 1]]
sample_data['wig'] = sample_data['Sample'] + ['.wig']
sample_data.rename(columns={'Raw Data':'depth'}, inplace=True)
sample_data[['wig', 'depth']].to_csv('mapping_wig_location_with_depth2.txt',sep='\t', index=False)

# 修改配置文件
sample_list = list(sample_data['wig'].values)
sample_list = ','.join(sample_list)

f_out = open('configure_file.txt', 'a')
with open('Dapars2_configure_file.txt', 'r') as input_file_handle:
    for line in input_file_handle.readlines():
        if line.startswith('Aligned_Wig_files'):
            aligned_wig = 'Aligned_Wig_files={0}\n'.format(sample_list)
            f_out.write(aligned_wig)
        else:
            f_out.write(line)
f_out.close()


# Step 2: Generate mapped reads files for all samples
cmmand_line = 'python DaPars2_Multi_Sample_Multi_Chr.py Dapars2_configure_file chrList.txt'
