import argparse
"""
parser = argparse.ArgumentParser(prog='PROG', description='this prog can fill your info')
parser.add_argument('input_name', help='input your name')
parser.add_argument('-ad', '--input_address', help='input your input_address')
parser.add_argument('-id', '--input_id', help='input your input_id')
args = parser.parse_args()


def test(input_name, input_address, input_id):
    if input_name:
        print('My name is {0}'.format(input_name))
    if input_address:
        print('My address is {0}'.format(input_address))
    if input_id:
        print('My ID is {0}'.format(input_id))
test(args.input_name, args.input_address, args.input_id)
"""

"""add_argument参数练习"""
"""
parser = argparse.ArgumentParser(prog='PROG', description='this prog can read your info')

# ============== 1. name or flags参数 ==============
# 添加一个必选参数
parser.add_argument('input_name', help='need your name')

# 添加一个可选参数
parser.add_argument('-name', '--input_name', help='need your name')

# ============== 2. action参数 ==============
# ArgumentParser 对象将命令行参数与动作相关联。这些动作可以做与它们相关联的命令行参数的任何事。action 命名参数指定了这个命令行参数应当如何处理
# action = 'store' - 存储参数的值。这是默认的动作，也就是在默认情况下，该参数返回的是在命令行中读取的值
# 例如input_name这个参数保留的命令行的值是"yangpei"
parser.add_argument('input_name', help='need your name', action='store')

# action = 'store_const' - 存储被 const 命名参数指定的值。 'store_const' 动作通常用在选项中来指定一些标志
# 例如当在命令行使用'-id'这个选项时，那么与之相关的动作就是将123作为'-id'选项的参数
# 而你只需指出-id这个选项即可，而不用传递任何参数，即使为该选项传递参数，也不会读取
parser = argparse.ArgumentParser(prog='PROG', description='this prog can read your info')
parser.add_argument('-id', help='need your id', action='store_const', const=123)
args = parser.parse_args()

print('参数是{0}'.format(args.id))

# action = 'store_true'/'store_false' - 这些是 'store_const' 分别用作存储 True 和 False 值的特殊用例。
parser = argparse.ArgumentParser(prog='PROG', description='this prog can read your info')
parser.add_argument('-id', help='id作为开关，如果存在则表示您需要id信息，会在系统中查询id信息', action='store_true')
args = parser.parse_args()
if args.id:
    print('当-id选项存在时，系统认为您需要id信息，在数据库中执行查找id')
else:
    print('当-id选项不存在时，系统认为您不需要id信息，在数据库中禁止查找id')
"""

# action = 'append' - 存储一个列表，并且将每个参数值追加到列表中。在允许多次使用选项时很有用。
parser = argparse.ArgumentParser(prog='PROG', description='this prog can read your info')
parser.add_argument('-id', help='id选项用来在数据库中进行匹配，'
                                'id选项需要传递进两次参数，第一个为区号，第二个为出生年份',
                    action='append')
args = parser.parse_args()
print('您输入的-id选型的参数是分别是{0} {1}'.format(args.id[0], args.id[1]))
print('查询到您的身份证号是411{0}{1}11114310'.format(args.id[0], args.id[1],))

