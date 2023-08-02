# 滑动窗口
sep = [1,2,3,4] 
# n表示窗口的长度，m表示从序列的哪个位置开始
def outter_function(n):
    def inner_function(m):
        return sep[m:m+n]
    return inner_function

a = outter_function(2)
b = a(1)
print(b)




