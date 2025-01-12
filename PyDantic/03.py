from pydantic import BaseModel, Field, model_serializer, field_serializer
from typing import List, Dict


class User(BaseModel):
    id: int
    name: str
    age: int
    is_admin: bool = Field(default=False)
    tags: List[str] = []

    # Customizing the serialization of the entire model
    @model_serializer
    def custom_serialize(self) -> Dict:
        # Change serialization to include only specific fields
        return {
            "user_id": self.id,
            "full_name": self.name,
            "details": f"{self.name}, Age {self.age}",
            "admin": self.is_admin,
            "tags": self.tags,
        }

    # Customizing the serialization of the `tags` field
    @field_serializer("tags")
    def serialize_tags(cls, tags: List[str]) -> str:
        return ", ".join(tags) if tags else "No Tags"


# Test the functionality
user = User(id=1, name="Alice", age=30, is_admin=True, tags=["developer", "writer"])

# Default `dict()` serialization
print("Default Dict:", user.dict())

# Custom serialization using model_serializer and field_serializer
print("Custom Serialization:", user.model_dump())
