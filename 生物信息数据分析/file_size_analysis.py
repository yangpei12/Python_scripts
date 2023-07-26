import os
import glob
import argparse
parser = argparse.ArgumentParser(
                    prog='Python Program',
                    description='This is a tool for caculate file size')
# 添加参数：
parser.add_argument('workDir', help='Need a Path')  # 位置参数
parser.add_argument('outDir', help='the txt which stats file size')  # 位置参数
# 解析参数：
args = parser.parse_args()

def file_size_stat(path, outDir):
    output = '{0}/file_size_stat.txt'.format(outDir)
    file_size_buffer = open(output, 'a')
    #使用os.walk函数进行遍历
    for root,dirs,files in os.walk(path):
        for sub_dir in dirs:
            globPath = os.path.join(root,sub_dir,'*.*')
            glob_result = glob.glob(globPath)
            for target_file in glob_result:
                file_size = os.path.getsize(target_file)
                result = '{0}\t{1:.0f}kb\n'.format(target_file, float(file_size)/1024)
                file_size_buffer.write(result)
    file_size_buffer.close()
file_size_stat(args.workDir, args.outDir)