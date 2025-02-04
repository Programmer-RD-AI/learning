import asyncio
import time


async def fetch_data_async(url):
    print(f"Fetching data from {url}...")
    await asyncio.sleep(2)  # Simulate a delay
    print(f"Finished fetching data from {url}")


async def asynchronous_main():
    urls = ["URL1", "URL2", "URL3"]
    tasks = [fetch_data_async(url) for url in urls]
    await asyncio.gather(*tasks)


print("\nAsynchronous Start")
start_time = time.time()
asyncio.run(asynchronous_main())
print(f"Asynchronous End (Duration: {time.time() - start_time:.2f} seconds)")
