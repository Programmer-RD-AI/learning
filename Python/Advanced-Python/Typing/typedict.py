from typing import TypedDict, ReadOnly


class User(TypedDict):
    id: ReadOnly[int]
    name: str
    email: str


user: User = {
    "id": 1,
    "name": "John Doe",
    "email": "gain@gmail.com",
}

user["id"] = 3  # Error: Cannot assign to read-only key "id"
User(**user)
