# Pydantic model for user registration
class User(BaseModel):
    username: str
    email: EmailStr
    password: str


# Pydantic model for an item (e.g., a product)
class Item(BaseModel):
    name: str
    description: str
    price: float
    tags: Optional[List[str]] = []


# Pydantic response model for consistency
class ItemResponse(Item):
    id: str

    class Config:
        orm_mode = True
