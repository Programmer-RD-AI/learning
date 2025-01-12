import requests
import asyncio


def send_request(url: str) -> int:
    print("Sending HTTP requests")
    response = requests.get(url)
    return response.status_code


async def send_async_request(url: str) -> int:
    return asyncio.to_thread(send_request, url)


async def main():
    status_code = await send_async_request("http://localhost:5000")
    return status_code
