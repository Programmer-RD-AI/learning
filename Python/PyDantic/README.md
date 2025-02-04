# Pydantic

## `enum.IntFlag`

Simply a type of enum that can be used for role selection or access level selection.

## `modes`: `('before', 'plain', 'wrap')`

### `before` mode

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    username: str

    @field_validator('username', mode='before')
    def strip_username(cls, value):
        return value.strip() if isinstance(value, str) else value

user = User(username="   JohnDoe   ")
print(user)  # username='JohnDoe'
```

### `plain` mode

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    age: int

    @field_validator('age', mode='plain')
    def check_age(cls, value):
        if value < 18:
            raise ValueError("Age must be 18 or older.")
        return value

user = User(age=20)  # Valid
print(user)          # age=20

# user = User(age=16)  # Raises: ValueError: Age must be 18 or older.
```

### `wrap` mode

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    email: str

    @field_validator('email', mode='wrap')
    def wrap_email_validation(cls, validator, values, config, field):
        print("Before validation:", values)
        value = validator(values)  # Call the next step in the chain
        print("After validation:", value)
        if not value.endswith("@example.com"):
            raise ValueError("Email must be from example.com domain.")
        return value

user = User(email="test@example.com")
print(user)

# user = User(email="test@gmail.com")  # Raises: ValueError: Email must be from example.com domain.
```

### Key Differences Between Modes

| **Mode**   | **Timing**                 | **Access to Raw Values** | **Access to Processed Values** | **Chaining Other Validators** |
| ---------- | -------------------------- | ------------------------ | ------------------------------ | ----------------------------- |
| **before** | Before type validation     | Yes                      | No                             | No                            |
| **plain**  | After type validation      | No                       | Yes                            | No                            |
| **wrap**   | Wraps the entire lifecycle | Yes (if desired)         | Yes                            | Yes                           |

## from

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str
    full_name: str

    @classmethod
    def from_email(cls, email: str):
        name, domain = email.split("@")
        return cls(username=name, full_name=f"{name.capitalize()} from {domain.capitalize()}")

# Create a User from an email
user = User.from_email("john.doe@example.com")
print(user)
# Output: username='john.doe' full_name='John.doe from Example.com'
```

## Notes

- model_validator Vs. field_validator: The model validator give's you access to the entire data dictioanry, where as the field validator just gives you that specific field
- If the field_validator or model_validator are after then we dont need the class method since its a instnace method at that point
