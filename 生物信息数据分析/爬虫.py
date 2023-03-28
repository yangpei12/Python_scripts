from bs4 import BeautifulSoup
import requests
r = requests.request('get',url='https://www.baidu.com')
r.encoding = 'utf-8'
