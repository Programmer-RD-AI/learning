import enum
import hashlib
import re
from typing import Any
from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

VALID_PASSWORD_REGEX = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,}$")
VALID_NAME_REGEX = re.compile(r"^[a-zA-Z]{2,}$")


class Role(enum.IntFlag):
    Author = 1
    Editor = 2
    Admin = 4
    SuperAdmin = 8


class User(BaseModel):
    name: str = Field(examples=["Arjan"])
    email: EmailStr = Field(
        examples=["example@gmail.com"],
        description="THe email address of the user",
        frozen=True,
    )
    password: SecretStr = Field(
        examples=["Password123"],
        description="The password of the user",
    )
    role: Role = Field(
        default=None, description="The role of the user", examples=[1, 2, 3, 4]
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not VALID_NAME_REGEX.match(v):
            raise ValueError("Name must contain at least 2 characters")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_user(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "name" not in v or "password" not in v:
            raise ValueError("User data must contain 'name' and 'password'")
        if v["name"].casefold() in v["password"].casefold():
            raise ValueError("Password cannot contain the name")
        if not VALID_PASSWORD_REGEX.match(v["password"]):
            raise ValueError(
                "Password must contain at least 1 uppercase letter, 1 lowercase letter, and 1 number"
            )
        v["password"] = hashlib.sha256(v["password"].encode()).hexdigest()


