import openai
from sqlalchemy.orm import Session
from crud import update_translation_task
from dotenv import load_dotenv
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def perform_translation(task: int, text: str, languages: list, db: Session):
    translations = {}
    for lang in languages:
        try:
            # response = openai.Completion.create(
                # engine="text-davinci-003",
                # prompt=f"Translate the following text to {lang}: {text}",
                # max_tokens=60,
                # api_key=OPENAI_API_KEY,
            # )
            translations[lang] = "Translated" # response.choices[0].text
        except Exception as e:
            translations[lang] = str(e)
        update_translation_task(db, task, translations)
    return translations
