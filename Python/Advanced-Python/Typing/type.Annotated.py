from typing import Annotated
from dataclasses import dataclass, field


@dataclass()
class Profile:
    name: str
    age: Annotated[int, lambda x: x > 0]
    jobs: list[str] = field(default_factory=list)

    def __setattr__(self, key, value):
        if field := self.__dataclass_fields__.get(key):
            if metadata := getattr(field.type, "__metadata__", None):
                assert metadata[0](value), f"Invalid Value {key!r}"
        super().__setattr__(key, value)


if __name__ == "__main__":
    p = Profile("Ranuga", 54, ["Software Enginnering"])
    print(p)
    p.age = -4
