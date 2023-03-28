
# a = iter(range(1, 10))
# active = True
# while active:
#     result = next(a)
#     print(result)
#     if a is None:
#         active = False

a = 'new york'
print('{0}'.format(a))

class dog():
    def __init__(self, leg, head):
        self.leg = leg
        self.head = head
    def __dogy__(self):
        return 'this dog has {0} legs {1} head'.format(self.leg, self.head)

my_dog = dog(4,1)
print(type(my_dog))

