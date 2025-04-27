# With __slots__


class FooBar:

    __slots__ = ("a", "b", "c")

    def __init__(self):

        self.a = 1

        self.b = 2

        self.c = 3


f = FooBar()

print(f.__dict__)  # AttributeError

print(f.__slots__)  # ('a', 'b', 'c')
