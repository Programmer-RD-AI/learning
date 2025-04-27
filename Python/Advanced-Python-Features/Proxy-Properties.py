from typing import Callable, Generic, TypeVar, ParamSpec, Self


P = ParamSpec("P")

R = TypeVar("R")

T = TypeVar("T")


class ProxyProperty(Generic[P, R]):

    func: Callable[P, R]

    instance: object

    def __init__(self, func: Callable[P, R]) -> None:

        self.func = func

    def __get__(self, instance: object, _=None) -> Self:

        self.instance = instance

        return self

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:

        return self.func(self.instance, *args, **kwargs)

    def __repr__(self) -> str:

        return self.func(self.instance)


def proxy_property(func: Callable[P, R]) -> ProxyProperty[P, R]:

    return ProxyProperty(func)


class Container:

    @proxy_property
    def value(self, val: int = 5) -> str:

        return f"The value is: {val}"


# Example usage

c = Container()

print(c.value)  # Returns: The value is: 5

print(c.value(7))  # Returns: The value is: 7
