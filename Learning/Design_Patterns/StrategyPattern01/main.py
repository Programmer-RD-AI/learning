import os
from typing import Any, Protocol
import mongo
import s3

class DataStore(Protocol):
    def put(self, key: str, value: Any) -> None:
        ...

    def get(self, key: str) -> Any:
        ...


