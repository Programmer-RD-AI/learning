from fastapi import (
    FastAPI,
    BackgroundTasks,
    HTTPException,
    Depends,
    Query,
    status,
    File,
    UploadFile,
)
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import shutil
import os
import uuid
from io import BytesIO
from .models import *
from .middleware import LoggingMiddleware

# OAuth2 authentication setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware to allow cross-origin requests
app.add_middleware(
    LoggingMiddleware,
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Set up Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="app/templates")

# Serve static files (e.g., CSS, images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Dependency: Token validation (simplified)
def get_current_user(token: str = Depends(oauth2_scheme)):
    if token != "valid-token":
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return {"user": "john_doe"}  # Simulating a user for the example


# Simulated database (a simple list)
fake_db_users = []
fake_db_items = []


# Background task function to log items added
def log_item_added(item_name: str):
    with open("log.txt", "a") as log_file:
        log_file.write(f"Item added: {item_name}\n")


# Home endpoint (using Jinja2 template)
@app.get("/", response_class=HTMLResponse)
async def home(request):
    return templates.TemplateResponse("index.html", {"request": request})


# Endpoint for user registration
@app.post("/register/", response_model=User)
async def register_user(user: User):
    fake_db_users.append(user)
    return user


# Endpoint to add an item (with background task)
@app.post("/add_item/", response_model=ItemResponse)
async def add_item(item: Item, background_tasks: BackgroundTasks):
    item_id = str(uuid.uuid4())
    fake_db_items.append(item.dict())
    background_tasks.add_task(log_item_added, item.name)
    return ItemResponse(id=item_id, **item.dict())


# Endpoint to get items with pagination and optional filters
@app.get("/items/", response_model=List[ItemResponse])
async def get_items(
    skip: int = Query(0, alias="page", ge=0),  # Pagination: start at 0
    limit: int = Query(10, le=100),  # Limit the number of results per page
    q: Optional[str] = None,  # Optional query parameter for search
):
    filtered_items = fake_db_items
    if q:
        filtered_items = [
            item
            for item in fake_db_items
            if q.lower() in item["name"].lower()
            or q.lower() in item["description"].lower()
        ]

    # Pagination: skip items and limit results
    start = skip * limit
    end = start + limit
    return [
        ItemResponse(id=str(i), **item)
        for i, item in enumerate(filtered_items[start:end])
    ]


# OAuth2 token endpoint (for simplicity, we just return a fixed token)
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    return {"access_token": "valid-token", "token_type": "bearer"}


# File upload endpoint (with file validation)
ALLOWED_FILE_TYPES = {"image/png", "image/jpeg"}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only PNG and JPEG are allowed."
        )

    file_location = f"app/static/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "file_location": file_location}


# Error handling endpoint (demonstrates custom HTTPException)
@app.get("/error/")
async def raise_error():
    raise HTTPException(status_code=404, detail="This is a custom error!")


# Serve static file (e.g., download image)
@app.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = f"app/static/{file_name}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")
