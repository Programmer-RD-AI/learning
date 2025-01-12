"""
# Decorator Design Pattern

Strcutral Pattern, that allows us to attach additional responsibilites to an object at run time.

The decorator pattern is used in both the Object Oriented and Functional paradigms.
"""

class UndecoratedObject:
    @staticmethod
    def get():
        return "UndecoratedObject"

class DecoratedObject:
    def __init__(self, undecorated):
        self.undecorated = undecorated

    def get(self):
        return self.undecorated.get().replace("Undecorated", "Decorated")

UNDECORATED = UndecoratedObject()
print(UNDECORATED.get())

DECORATED = DecoratedObject(UNDECORATED)
print(DECORATED.get())
