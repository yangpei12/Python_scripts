# 创建一个元组列表
listed_tuple = [('CHA', 'BJ'), ('JPN', 'TK'), ('USA', 'WD'), ('FRA', 'PA'), ('UK', 'LD')]

# 元组拆包的关键是，元组内的元素个数要与被赋值变量的个数一致，也就是 “ 平行赋值 ”
# 如果元组的存储方式为可迭代，那么可使用遍历进行逐个拆包
for country, capital in listed_tuple:
    print('The country name is %s, its captial is %s' %(country, capital))

