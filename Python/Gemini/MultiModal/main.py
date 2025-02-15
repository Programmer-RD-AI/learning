import asyncio
from google import genai
from dotenv import load_dotenv
import os


async def send_audio_video(session, audio_file, video_file):
    # Read audio file (raw 16-bit PCM, 16kHz)
    with open(audio_file, "rb") as af:
        audio_data = af.read()
    # Read video file (raw video data; adjust as needed)
    with open(video_file, "rb") as vf:
        video_data = vf.read()

    # Send audio data as a realtime input message.
    await session.send(input={"realtimeInput": {"media_chunks": [audio_data]}})
    print("Audio data sent.")

    # Send video data similarly.
    await session.send(input={"realtimeInput": {"media_chunks": [video_data]}})
    print("Video data sent.")


async def main():
    # Load environment variables
    load_dotenv()

    # Initialize the client with your API key
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"), http_options={"api_version": "v1alpha"}
    )
    model_id = "gemini-2.0-flash-exp"

    # Configure the session to support text and audio responses.
    config = {
        "responseModalities": ["TEXT", "AUDIO"],
        "speechConfig": {
            "voiceConfig": {
                "prebuiltVoiceConfig": {
                    "voiceName": "Kore"  # Example voice name; see docs for available voices
                }
            }
        },
    }

    # Establish a WebSocket session with the Gemini server.
    async with client.aio.live.connect(model=model_id, config=config) as session:
        # --- Step 1: Send a text message ---
        text_message = input("Enter text input: ")
        await session.send(
            input={
                "clientContent": {
                    "turns": [{"parts": [{"text": text_message}], "role": "user"}],
                    "turnComplete": True,
                }
            }
        )

        # --- Step 2: Optionally, send audio and video data ---
        audio_file = "audio.raw"
        video_file = "video.raw"
        await send_audio_video(session, audio_file, video_file)

        # --- Step 3: Receive and process responses ---
        async for response in session.receive():
            if response.text:
                print("Response text:", response.text)
            # Additional processing for audio/video blobs can be added here.


if __name__ == "__main__":
    asyncio.run(main())
