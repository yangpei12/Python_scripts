# 迭代器协议需要两部分__iter__()和__next__方法。

# 教材中的实际方法
class Node:
    def __init__(self, value):
        self._value = value
        self._children = []

    def add_child(self, node):
        self._children.append(node)

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def __iter__(self):
        return iter(self._children)

if __name__ == '__main__':
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)
    for i in root:
        print(i)



# class ITERS:
#     def __init__(self):
#         self.obj = []
#
#     def add_nums(self, num):
#         self.obj.append(num)
#
#     def __iter__(self):
#         return iter(self.obj)
#
#     def __next__(self):
#         return next(self.obj)
#
# if __name__ == '__main__':
#     its = ITERS()
#     its.add_nums(1)
#     its.add_nums(2)
#     its.add_nums(3)
#
#     print(next(its))



