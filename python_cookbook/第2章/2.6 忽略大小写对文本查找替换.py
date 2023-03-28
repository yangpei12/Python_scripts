import re
from calendar import month_abbr

"""
text = 'UPPER PYTHON, lower python, Mixed Python'
def matchcase(word):
    def replace(m):
        a = m.group()
        if a.isupper():
            return word.upper()
        elif a.islower():
            return word.lower()
        elif a[0].isupper():
            return word.capitalize()
        else:
            return word
    return replace

result = re.sub('python', matchcase('snake'), text, flags=re.IGNORECASE)
print(result)
"""

text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
datepat = re.compile(r'(\d+)/(\d+)/(\d+)')

def change_date(m):
    mon_name = month_abbr[int(m.group(1))]
    return '{} {} {}'.format(m.group(2), mon_name, m.group(3))

result = datepat.sub(change_date, text)
print(result)


