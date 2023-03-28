#正则表达
#匹配字符串 ‘我最喜欢的 foods are pizza 186’

import re
result = re.match(r'(.+)','我最喜欢的foods are pizza 186')
print(result.group(1))







'''
#匹配分组可以进行目标字符串中部分信息的提取
#匹配邮箱 yp2003qq@gmail.com
import re
result = re.match(r'(\w+)@(gmail|qq|163|126)\.(com|cn)$','yp2003qq@qq.com')
print(result)
print(result.group(1))
print(result.group(2))
print(result.group(3))
'''
'''
s='aaa111aaa,bbb222,333ccc,444ddd444,555eee666,fff777ggg'
#找出中间夹有数字的字母
import re
result = re.findall(r'[a-z]+\d+[a-z]+',s)
print(result)
#找出被中间夹有数字的前后同样的字母
result = re.findall('(?P<>)')
'''

import re
line ='192.168.0.1 25/Oct/2012:14:46:34 "GET /api HTTP/1.1" 200 44 "http://abc.com/search" "Mozilla/5.0"'
reg = re.compile('^(?P<remote_ip>[^ ]*) (?P<date>[^ ]*) "(?P<request>[^"]*)" '
                 '(?P<status>[^ ]*) (?P<size>[^ ]*) "(?P<referrer>[^"]*)" '
                 '"(?P<user_agent>[^"]*)"')
print(reg.findall(line))
'''
regMatch = reg.match(line)
linebits = regMatch.groupdict()
print (linebits)
for k, v in linebits.items() :
    print (k+": "+v)

s = "GET / api HTTP /1.1 "
pattern = re.compile(r'(?:.+)')
result = pattern.findall(s)
print(result)
'''




