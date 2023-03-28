import re

# 匹配以choices开头的URL
choices = ['http:', 'ftp:', 'https:']
url = 'https://www.python.org'

# 使用正则表达式提取前缀
pattern = re.compile(r'(http:| ftp: |https:).+')
prefix = pattern.match(url).group(1)
print(prefix)

# 结合startswith()
print(url.startswith(prefix))