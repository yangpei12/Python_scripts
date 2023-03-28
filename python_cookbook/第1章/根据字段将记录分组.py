from operator import itemgetter
from itertools import groupby
from collections import defaultdict
rows = [
 {'address': '5412 N CLARK', 'date': '07/01/2012'},
 {'address': '5148 N CLARK', 'date': '07/04/2012'},
 {'address': '5800 E 58TH', 'date': '07/02/2012'},
 {'address': '2122 N CLARK', 'date': '07/03/2012'},
 {'address': '5645 N RAVENSWOOD', 'date': '07/02/2012'},
 {'address': '1060 W ADDISON', 'date': '07/02/2012'},
 {'address': '4801 N BROADWAY', 'date': '07/01/2012'},
 {'address': '1039 W GRANVILLE', 'date': '07/04/2012'},
]

# 首先进行排序
rows.sort(key=itemgetter('date'))

# 对分组对象进行分组
grouped = groupby(rows, key=itemgetter('date'))

# 创建字典，格式是‘date’作为键，分组信息作为值。因此该字典是一个一键多值字典
d = defaultdict(list)

for date, infos in grouped:
 for info in infos:
  d[date].append(info)
print(d['07/01/2012'])


"""
# 排序练习
sorted_itemgetrter = sorted(rows, key=itemgetter('date'))
sorted_lambda = sorted(rows, key=lambda x: x['date'])
"""
