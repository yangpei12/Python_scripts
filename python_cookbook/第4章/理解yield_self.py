class DOG():
    def __init__(self, legs):
        self.legs = legs

    # def __iter__(self):
    #     return iter(self.legs)

    def dog(self):
        active = True
        while active:
            yield self


dog = DOG([1,2,3,4])
print(dog)
# result = dog.dog()
# for temp in result:
#     print(temp)