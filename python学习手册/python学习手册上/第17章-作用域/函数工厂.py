"""
def maker(N):
    def action(X):
        return X ** N
    return action
f = maker(2)
print(f(2))


def maker(N=2):
    return lambda x : x**N
f = maker(3)
print(f(4))
"""

def f1():
    x = 88
    def f2(x=x):
        print(x)
    f2()
print(f1())

