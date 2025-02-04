from enum import auto, IntFlag
from typing import Any
from pydantic import BaseModel, EmailStr, Field, SecretStr, ValidationError


class Role(IntFlag):
    Author = auto()
    Editor = auto()
    Developer = auto()
    Admin = Author | Editor | Developer


class User(BaseModel):
    name: str = Field(examples=["Arjan"])
    name: EmailStr = Field(
        examples=["example@gmail.com"],
        description="The email address of the user",
        frozen=True,  # only set it but cant change it afterwards
    )
    password: SecretStr = Field(
        default=["Password123"], description="The password of the user"
    )
    role: Role = Field(default=None, description="The role of the user")


def validate(data: dict[str, Any]) -> None:
    try:
        user = User.model_validate(data)
        print(user)
    except:
        raise ValidationError


def main():
    good_data = {
        "name": "Arjan",
        "email": "example@gmail.com",
        "password": "Password123",
    }
