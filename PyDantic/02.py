import enum
import hashlib
import re
from typing import Any, Self
from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
    field_serializer,
    model_serializer,
)

VALID_PASSWORD_REGEX = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,}$")
VALID_NAME_REGEX = re.compile(r"^[a-zA-Z]{2,}$")


class Role(enum.IntFlag):
    User = 0
    Author = 1
    Editor = 2
    Admin = 4
    SuperAdmin = 8


class User(BaseModel):
    name: str = Field(examples=["Arjan"])
    email: EmailStr = Field(
        examples=["user@arjancodes.com"],
        description="The email address of the user",
        frozen=True,
    )
    password: SecretStr = Field(
        examples=["Password123"],
        description="The password of the user",
    )
    role: Role = Field(
        description="The role of the user",
        examples=[1, 2, 3, 4],
        default=0,
        validate_default=True,
    )

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        if not VALID_NAME_REGEX.match(v):
            raise ValueError("Name must contain at least 2 characters")
        return v

    @model_validator(mode="after")
    def validate_user_post(self, v: Any) -> Self:
        if self.role == Role.Admin and self.name != "Arjan":
            raise ValueError("Only Arjan can be an Admin")
        return self

    @field_serializer("role", when_used="json")
    @classmethod
    def serialize_role(cls, v: Role) -> int:
        return v.value

    @model_serializer(mode="wrap", when_used="json")
    def serialize_user(self, serializer, info) -> dict[str, Any]:
        if not info.include and not info.exclude:
            return {"name": self.name, "role": self.role.name}
        return serializer(self)
