import typing

if typing.TYPE_CHECKING:
    import datetime

def func(dt: datetime.datetime) -> None:
    print(dt)
