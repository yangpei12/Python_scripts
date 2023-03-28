"""
X = 99 # 全局变量: 并未被使用
def f1():
    X = 88 # 封闭的 def 局部变量
    def f2():
        print(X) # 嵌套def中的引用
    f2()
print(f1()) # Prints 88: 封闭的 def 局部变量
"""

X = 99
def f1():
    X = 88
    def f2():
        print(X)
    return f2  # 注意这里仅仅返回的是函数名
action = f1() # 调用f1()实际上返回的是f2
action() # 这个地方实际上是f2()，也即是调用了f2函数