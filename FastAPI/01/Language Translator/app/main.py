from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import schemas
import crud
from database import get_db, engine
import models

models.Base.metadata.create_all(bind=engine)
app = FastAPI()

# Setup templates

templates = Jinja2Templates(directory="templates")


@app.get("/index", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/translate", response_model=schemas.Task)
def translate(request: schemas.TranslationRequest):
    task = crud.create_translation_task(get_db.db, request.text, request.languages)
    # BackgroundTasks.create_bac
    background_tasks.add_task(
        perform_translation, task.id, request.text, request.languages, get_db.db
    )
    return {"task_id": task.id}
