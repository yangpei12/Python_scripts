from collections import deque

class linehistory:
    def __init__(self, lines, histlen=3):
        self.lines = lines
        self.history = deque(maxlen=histlen)

    def __iter__(self):
        for lineno, line in enumerate(self.lines, 1):
            self.history.append((lineno, line))
            yield line

    def clear(self):
        self.history.clear()

path = r"D:\pycharm\project\测试用文本文件\somefile.txt"

f = open(path)
f_iterable = iter(f)
print(next(f_iterable))

"""
    lines = linehistory(f)
    for line in lines:
        if 'python' in line:
            for lineno, hline in lines.history:
                print('{0}:{1}'.format(lineno, hline), end='')
"""