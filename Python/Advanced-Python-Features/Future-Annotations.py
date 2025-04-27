from __future__ import annotations
from typing import Self


class Foo:

    def bar(self) -> Self: ...


class Foo:

    def bar(self) -> Foo:  # Works now!
        ...
