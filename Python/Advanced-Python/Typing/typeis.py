from typing import TypeGuard, TypeIs


class Dog:
    def bark(self) -> None:
        print("Woof")


class Cat:
    def meow(self) -> None:
        print("Meow")


def is_dog(animal: object) -> TypeIs[Dog]:
    return isinstance(animal, Dog)


def speak(animal: Dog | Cat) -> None:
    if is_dog(animal):
        animal.bark()
    else:
        animal.meow()


speak(Cat())
