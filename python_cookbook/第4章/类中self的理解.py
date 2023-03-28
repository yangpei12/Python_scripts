"""
class Test():
    def __init__(self, value):
        self.value = value

    # def __iter__(self):
    #     return iter(self.value)

    # def __next__(self):
    #     return

    def vi(self):
        return self

test = Test([1,2,3,4])
result = test.vi()
print(result)
# for temp in result:
#     print(temp)
"""

class Test():
    def __init__(self, value):
        self.value = value

    # def __iter__(self):
    #     return self

    def __next__(self):
        self.value += 1
        return self.value

    def view(self):
        return self.value

test = Test(2)
print(test.view())
# for temp in test:
#     print(temp)