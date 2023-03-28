# #join()函数测试
# a = ['1','2','3']
# b = ''.join(a)
# print(b)      #123
# print(type(b)) #<class 'str'>
# a = ['456']
# b = ''.join(a)
# print(b)     #456
# print(type(b))  #<class 'str'>

#isinstance()函数测试
# a = ['1','2','3']
# if isinstance(a,str):  #判断目标对象的类型是否属于第二个参数表现的类型。如果是返回值为True
#     print('Yes')
# else:
#     print('No')     #No
# if isinstance(a,list):
#     print('Yes')    #Yes
# else:
#     print('No')

#split()函数测试
# a = 'asd dsd gfg aaf'
# b = a.split(' ') #以传入参数为分隔将一个字符串分割为多个字符串，返回值是分割后各字符串的列表
# print(b)    #['asd', 'dsd', 'gfg', 'aaf']
# c = ''.join(b)   #split()函数,可与join()函数搭配，将列表中的字符串合成一个字符串，且无空格
# print(c)    #asddsdgfgaaf

#正则表达测试
# import re
# with open('生物信息学测试\HBA_RAT.FastA.txt') as file:
#     newFile = file.read()
# if isinstance(newFile,str):
#     #提取第一行
#     pattern = re.compile(r'(>.+3)')
#     result = pattern.match(newFile)
#     result_2 = pattern.findall(newFile)
#     b = ''.join(result_2)
#     print(result)
#     print(b.count(''))
# else:
#     print('No')

#测试非捕获组
# import re
# s = '5566 5555 6655 55566'
# pattern = re.compile(r'(?:\b5{2})(?:6+)\b') #加入边界后它会像match一样。依据空格为分界，将每个子串
# result = pattern.findall(s)               #视为整体一个一个筛查，直到找到一个子串完全匹配表达式
# print(result) #['5566']                   #并将其整体都提取出来。若匹配的字符串是一个字符串的一部分
#                                          #则findall不认为该字符串符合规则。必须整体匹配才行
#
# pattern = re.compile(r'(?:5{2})(?:6+)') #如果不加边界，findall只要找到符合匹配表达式的字符串就将
# result = pattern.findall(s)            #其提取出来。不论该字符串是否是一个字符串的一部分。
# print(result)#['5566', '5566']

# #测试findall与match的区别
# import re
# s = 'abcab123sd4512ab'
# pattern = re.compile(r'([a-z]+)([0-9]+)')
# result_1 = pattern.findall(s)
# print(result_1)
# #[('abcab', '123'), ('sd', '4512')]
# #findall 会从字符串的左边开始，搜寻符合表达式的字符串，匹配一个返回一个，一直搜寻到字符串结尾。
# #对于括号括起来的分组，findall会将两个经过匹配分组的表达式割裂开来，分别匹配[a-z]的字符串与匹配[0-9]的字符串，
# #看作两个字符串，并放在元组中。形成('abcab', '123')的结果。然后继续向后搜索，重复上述模式
# result_2 = pattern.match(s)
# print(result_2)
# #<re.Match object; span=(0, 8), match='abcab123'>
# #match函数从字符串的开头开始，只要遇到符合表达式的结果就返回，不再往后匹配。
# #而match函数会将匹配分组的表达式看作一个整体，将匹配表达式的最终合并在一起。
#
# #findall函数会搜索整个字符串，并将全部符合表达式的字符串全部提取出来，不过有n个括号，它会将返回n个
# #字符串，并将其返回一个元组中
# #match函数会从头开始匹配表达式，一旦找到一个符合表达式的字符串，就返回结果。
# #即使有n个括号，它最终还是会将匹配的n个字符串合为一体。

# #给每行字符串加索引
# with open('文件操作练习','r',encoding='utf-8') as file:
#     lines = file.readlines()
#     i = 0
#     for line in lines:
#         b = line.rstrip()
#         b += ' ' + '#' + str(i)
#         i+=1
#         print(b)


# #正则练习
# #取出192.168.1.100
# import re
# s = 'inet 192.168.1.100 netmask 255.255.255.0 broadcast 192.168.1.255'
# pattren = re.compile(r'[a-z]+\s(.+?\s).+') #防止贪婪
# result = pattren.findall(s)
# print(result)

# #sys模块练习
# import sys
# var_1 = sys.argv[1]
# var_2 = sys.argv[2]
# print(var_1)
# print(var_2)

#输出特定长度 [i:i+n]的字符串
seq = 'PRQTEINSEQWENCE'
for i in range(len(seq)):
    lst =[]
    for temp in seq[i:i+4]:     #此处的技巧需多加揣摩
        lst.append(temp)
        if len(lst)==4:
            print(' '.join(lst))

#用类进行数字计算
with open('numcaculate.txt') as file:
    lines = file.readlines()
    lst_num =[]
    for line in lines:
        lst_num.append(float(line.rstrip())) #此处需用float函数将字符串转换为浮点数，否则
    print(lst_num)                           #sum()函数会报错，不支持str相加

class Data_Caculaton():                #该类对"传入文件"中的数据进行处理
    def __init__(self,nums):
        lst=[]
        for temp in nums:
            self.temp = temp
            lst.append(self.temp)
        self.lst = lst
        print(self.lst)
    def num_Add(self):
         print(sum(self.lst))

    def max_num(self):
        print(max(self.lst))


num = Data_Caculaton(lst_num)
num.num_Add()
num.max_num()

#练习，传出不定数量的参数用 *argvs 形式。
class Data_Caculaton():                #该类进行"传入多个数据"进行简单的数据计算
    def __init__(self,*nums):    #在形参用“*argvs”表示后。变量argvs就成为了一个包含传入参数的“列表”
        lst = []
        for temp in nums:        #要使用argvs列表中的元素，需用遍历的形式取出
            self.temp = temp
            lst.append(self.temp)
        self.lst = lst
        print(self.lst)
    def num_Add(self):
         print(sum(self.lst))

    def max_num(self):
        print(max(self.lst))


num = Data_Caculaton(1,2,3,4)
num.num_Add()
num.max_num()