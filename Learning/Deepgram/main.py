from deepgram import Deepgram
import asyncio

# Replace 'your_api_key' with your Deepgram API key
DEEPGRAM_API_KEY = "your_api_key"
dg_client = Deepgram(DEEPGRAM_API_KEY)


async def transcribe_audio():
    audio_path = "path_to_audio_file.mp3"
    with open(audio_path, "rb") as audio:
        response = await dg_client.transcription.prerecorded(
            {"buffer": audio, "mimetype": "audio/mpeg"}  # Adjust based on the file type
        )
    print(response["results"]["channels"][0]["alternatives"][0]["transcript"])


# Run the asynchronous function
asyncio.run(transcribe_audio())

import websockets


async def transcribe_live():
    url = "wss://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
    }

    async with websockets.connect(url, extra_headers=headers) as ws:
        # Send audio data here (e.g., from a microphone)
        # Example: await ws.send(audio_chunk)
        async for message in ws:
            print(message)


# asyncio.run(transcribe_live())
response = await dg_client.transcription.prerecorded(
    {
        "buffer": audio,
        "mimetype": "audio/mpeg",
        "keywords": ["Deepgram", "Python"],
        "language": "en",
        "diarize": True,
    }
)
try:
    response = await dg_client.transcription.prerecorded(
        {"buffer": audio, "mimetype": "audio/mpeg"}
    )
except Exception as e:
    print(f"Error: {e}")
