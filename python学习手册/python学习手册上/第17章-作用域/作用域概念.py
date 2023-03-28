# x = 99  # x为全局变量
# def func(y):
#     z = x +y # z为局部变量
#     return z

x = 88
def func():
    global x
    x = 99
    return x

print(func()+1)

