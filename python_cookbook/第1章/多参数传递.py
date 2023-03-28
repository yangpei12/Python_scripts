
def test_function(a, b, c):
    return a++b+c

d = [1, 2, 3, 4, 5]
a = test_function(*d)
print(a)