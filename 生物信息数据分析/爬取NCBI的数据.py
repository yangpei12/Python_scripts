import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd

os.chdir("E:\售后")
sample_list = pd.read_excel("GSE21138\GSE21138.xlsx",sheet_name=0)
samples = [x for x in sample_list['Sample']]
output = open("GSE21138\GSE21138_sample.txt",'a')

for gsm in ['GSM528831']:
    url = '{0}{1}'.format("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=",gsm)
    r = requests.get(url)
    r.encoding = r.apparent_encoding
    html = r.text
    # 为beautiful显式指定一个html解析器，不同系统所使用html解析器不同
    soup = BeautifulSoup(html,"html.parser")
    #print(css_soup.string)
    print(soup.select('td style'))

    #s = '{0}\t{1}'.format(gsm,soup[0])
    #output.write('{0}\t{1}{2}'.format(gsm,soup[0],"\n"))

