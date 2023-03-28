import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd
url = '{0}{1}'.format("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=","GSM528831")
r = requests.get(url)
r.encoding = r.apparent_encoding
html = r.text

soup = re.findall(r'stage of illness .+',html)
print(soup)