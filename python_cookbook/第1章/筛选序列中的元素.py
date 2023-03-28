"""
# 使用列表推导式筛选序列中的元素，通常推导式中包含判断语句。
# 如果遇到原始数据非常大的情况，那么使用生成器表达式会更加合适
mylist = [1, 4, -5, 10, -7, 2, 3, -1]
pos = (x for x in mylist if x > 0)
print(pos)
for i in pos:
    print(i)

# 使用filer函数进行数据筛选
"""

addresses = [
 '5412 N CLARK',
 '5148 N CLARK',
 '5800 E 58TH',
 '2122 N CLARK'
 '5645 N RAVENSWOOD',
 '1060 W ADDISON',
 '4801 N BROADWAY',
 '1039 W GRANVILLE',
]
counts = [0, 3, 10, 4, 1, 7, 6, 1]
# more5_1 = [x > 5 for x in counts]
# more5_2 = [x for x in counts if x > 5 else 0]
# more5_3 = [x if ]
# clip_neg = [n if n > 5 else 0 for n in counts]
# clip_neg = [n for n in counts if n > 5 else 0]

clip_neg = []
for n in counts:
    if n > 5:
        clip_neg.append(n)
    else:
        n = 0
        clip_neg.append(n)

print(clip_neg)