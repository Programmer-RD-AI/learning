# FastAPI

## Middleware

```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
import time

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Log the request
        start_time = time.time()
        print(f"Request started: {request.method} {request.url}")

        # Call the next middleware or the request handler
        response = await call_next(request)

        # Log the response and the time taken to process the request
        process_time = time.time() - start_time
        print(f"Request finished: {request.method} {request.url} - Time taken: {process_time:.4f} seconds")

        # Return the response
        return response

# Initialize FastAPI app
app = FastAPI()

# Add the custom logging middleware
app.add_middleware(LoggingMiddleware)

# Example endpoint
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
```

# response_model

If you don't specify a response_model, the behavior changes in the following ways:

- No Validation on Response
- No OpenAPI Schema for Response
- Manual Serialization

## Without response_model

- No response validation is done.
- No response schema is shown in OpenAPI docs.
- You need to handle serialization manually.

## With response_model

- Response validation and serialization are handled automatically by FastAPI.
- OpenAPI documentation includes detailed response schema.
- You get automatic conversion to JSON with structure validation.

# Synchronous

When you define an endpoint using a synchronous function, FastAPI processes one request at a time for that function. If a request is being processed, all other requests for that route (or even other routes) will be blocked until the current request finishes.

Synchronous functions block the execution of the application. If you're doing something time-consuming (like making a database query, performing I/O operations, or waiting for an external API response), the server will wait for the operation to complete before it can process any further requests.

# Asynchronous

When you define an endpoint using an async def function, FastAPI can handle multiple requests concurrently. When the server encounters an I/O-bound operation (such as waiting for a response from a database, external API, or file I/O), it doesn’t block the entire process. Instead, it pauses the current request and moves on to handle other incoming requests. Once the I/O operation is complete, it resumes the current request.

Asynchronous functions allow FastAPI to efficiently handle many requests at once by releasing the CPU to do other work while waiting for I/O operations to complete. This is particularly beneficial in applications that perform many external requests, database calls, or have long-running operations.

Asynchronous functions improve performance for I/O-bound tasks by allowing the server to handle multiple requests concurrently while waiting for resources (e.g., database, external APIs).

# Asynchronous vs. Synchronous

| Feature            | **Synchronous (`def`)**                                                 | **Asynchronous (`async def`)**                                                       |
| ------------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Blocking**       | Blocks the server while waiting for I/O operations.                     | Does not block the server; handles other requests while waiting.                     |
| **Performance**    | Less efficient for I/O-bound tasks (e.g., database queries, API calls). | More efficient for I/O-bound tasks, improves performance with many concurrent users. |
| **Concurrency**    | Can handle only one request at a time per thread.                       | Can handle multiple requests concurrently using `asyncio`.                           |
| **Ideal Use Case** | Simple applications or CPU-bound tasks.                                 | Web applications that perform a lot of I/O-bound operations.                         |
| **Syntax**         | `def function_name():`                                                  | `async def function_name():`                                                         |

# Parameters

## Query Parameter

`GET /items/?skip=5&limit=20`

```python
@app.get("/items/")
async def read_items(skip: int = Query(0), limit: int = Query(10)):
    return {"skip": skip, "limit": limit}
```

## Path Parameters

`GET /items/42`

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

## Body Parameters

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str
    price: float

@app.post("/items/")
async def create_item(item: Item):
    return item
```

Request Body:

```json
{
  "name": "Item A",
  "description": "This is a great item.",
  "price": 10.5
}
```

## Form Parameters

```python
from fastapi import Form

@app.post("/login/")
async def login(username: str = Form(...), password: str = Form(...)):
    return {"username": username}
```

## File Parameters

```python
from fastapi import File, UploadFile

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}
```

```text
Content-Disposition: form-data; name="file"; filename="example.txt"
```

## Header Parameters

```python
from fastapi import Header

@app.get("/headers/")
async def read_header(user_agent: str = Header(...)):
    return {"User-Agent": user_agent}
```

## Cookie Parameters

```python
from fastapi import Cookie

@app.get("/cookies/")
async def read_cookie(session_id: str = Cookie(...)):
    return {"session_id": session_id}
```
