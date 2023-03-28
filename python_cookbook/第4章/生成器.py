"""
class listDemo:
    def __init__(self):
        self.__date=[]
        self.__step = 0
    # def __next__(self):
    #     if self.__step <= 0:
    #         raise StopIteration
    #     self.__step -= 1
    #     #返回下一个元素
    #     return self.__date[self.__step]
    def __iter__(self):
        #实例对象本身就是迭代器对象，因此直接返回 self 即可
        return self
    #添加元素
    def __setitem__(self,key,value):
        self.__date.insert(key,value)
        self.__step += 1
mylist = listDemo()
mylist[0]=1
mylist[1]=2
for i in mylist:
    print (i)
"""

class MyNumbers:
    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        x = self.a
        self.a += 1
        return x
myclass = MyNumbers()
myiter = iter(myclass)
print(next(myiter))


