import json, time, os
from google import genai

from google.genai import types
from pydantic import BaseModel
from typing import Optional

# load environment variables
from dotenv import load_dotenv

load_dotenv()

# set up gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = "gemini-2.0-flash"

video_file = client.files.upload(file="video.mp4")

while video_file.state.name == "PROCESSING":
    print("processing video...")
    time.sleep(5)
    print("video file name:")
    print(video_file.name)
    video_file = client.files.get(name=video_file.name)

# video_file = client.files.get(name="files/file123")


# set up pydantic models for companies and themes
class Company(BaseModel):
    name: str
    bullish_or_bearish: str
    why: str


extract_ideas_from_video_prompt = """
I have attached a YouTube video. Explain everything mentioned in the video
"""

# count the tokens in the prompt and file
print(
    client.models.count_tokens(
        model=model, contents=[video_file, extract_ideas_from_video_prompt]
    )
)

# send the prompt and file to gemini
result = client.models.generate_content(
    model=model,
    contents=[video_file, extract_ideas_from_video_prompt],
    config=types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=list[Company]
    ),
)

print(json.loads(result.text))
