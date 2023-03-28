import re

# 使用正则表达式进行切割
line = 'asdf fjdk; afed, fjek,asdf, foo'
new_line = re.split(r'[\s;,]\s*', line)
new_line = re.split(r'[\s;,]', line)
print(new_line)