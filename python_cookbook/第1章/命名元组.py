from collections import namedtuple
Subscriber = namedtuple('Apple', ['addr', 'joined'])

sub = Subscriber('jonesy@example.com', '2012-10-19')

print(sub)
