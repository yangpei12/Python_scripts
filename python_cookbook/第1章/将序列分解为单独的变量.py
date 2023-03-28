# 在进行分解操作时可使用特殊语法丢弃特定值
data = ['ACME', 50, 91.1, (2012, 12, 21)]
_, num, price, _ = data
print(num)
print(price)

#
records = [('foo', 1, 2), ('bar', 'hello'), ('foo', 3, 4)]
for tag, *args in records:
    print(tag)


