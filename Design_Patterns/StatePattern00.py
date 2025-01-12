from typing import Protocol


class LightState(Protocol):
    def switch(self, bulb) -> None:
        ...

class OnState:
    def switch(self, bulb) -> None:
        bulb.state = OffState()

class OffState:
    def switch(self, bulb) -> None:
        bulb.state = OnState()

class Bulb:
    def __init__(self) -> None:
        self.state = OnState()

    def switch(self) -> None:
        self.state.switch(self)
