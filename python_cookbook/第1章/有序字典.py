#
"""
a = [1, 2, 3, 4]
b = ['a', 'b', 'c', 'd']
c = {'num': a, 'alpha': b}
print(c['num'])

def dedupe(items):
    seen = set()
    for item in items:
        if item not in seen:
            #yield item
            seen.add(item)
    return seen
a = [1, 5, 2, 1, 9, 1, 5, 10]
print(list(dedupe(a)))
"""

words = ['look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
 'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around', 'the',
 'eyes', "don't", 'look', 'around', 'the', 'eyes', 'look', 'into',
 'my', 'eyes', "you're", 'under']
# 初始化一个计数字典，键为单词，值为次数
counts = {}
for key in words:
    counts[key] = 0
print(counts)

# 开始计数，若单词存在于字典的键中则次数加1
for key in words:
    if key in counts.keys():
        counts[key] += 1
    else:
        counts[key] = 1
print(counts)

morewords = ['why', 'are', 'you', 'not', 'looking', 'in', 'my', 'eyes', 'eyes']
for word in morewords:
    counts[word] += 1
print(counts)