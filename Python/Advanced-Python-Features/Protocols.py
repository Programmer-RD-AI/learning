from typing import Protocol


class Quackable(Protocol):

    def quack(
        self,
    ) -> None: ...  # The ellipsis indicates this is just a method signature


class Duck:

    def quack(self):
        print("Quack!")


class Dog:

    def bark(self):
        print("Woof!")


def run_quack(obj: Quackable):

    obj.quack()


run_quack(Duck())  # Works!

run_quack(Dog())  # Fails during TYPE CHECKING (not runtime)
