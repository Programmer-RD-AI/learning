from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def basic():
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=["How does AI work?"]
    )

    print(response.text)


def streaming():
    response = client.models.generate_content_stream(
        model="gemini-2.0-flash", contents=["Explain how AI works"]
    )
    for chunk in response:
        print(chunk.text, end="")


def configuration():
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["Explain how AI works"],
        config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1),
    )
    print(response.text)


def system_info():
    sys_instruct = "You are a cat. Your name is Neko."
    client = genai.Client(api_key="GEMINI_API_KEY")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction=sys_instruct),
        contents=["your prompt here"],
    )
    return response
