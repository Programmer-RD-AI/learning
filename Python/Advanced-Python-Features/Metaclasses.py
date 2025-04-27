class DoubleAttrMeta(type):

    def __new__(cls, name, bases, namespace):

        new_namespace = {}

        for key, val in namespace.items():

            if isinstance(val, int):

                val *= 2

            new_namespace[key] = val

        return super().__new__(cls, name, bases, new_namespace)


class MyClass(metaclass=DoubleAttrMeta):

    x = 5

    y = 10


print(MyClass.x)  # 10

print(MyClass.y)  # 20
