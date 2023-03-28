import re
l ='192.168.0.1 25/Oct/2012:14:46:34 "GET / api HTTP /1.1" 200 44 "http://abc.com/search" "Mozilla/5.0"'
pattern = re.compile(r'(?P<ip>.+) (?P<date>.+) (?P<api>".+") (?P<num_1>.+) (?P<num_2>.+) (?P<http>".+") (?P<tool>".+")')
result = pattern.findall(l)
print(result)

result = pattern.match(l)
result = result.groupdict()
for k,v in result.items():
    print(k + ' ' + v)
#该例题的注意点。由字符串可知，各个子串之间有空格，则使用match函数时各个表达式之间也要有空格，
#与字符串保持一致。
