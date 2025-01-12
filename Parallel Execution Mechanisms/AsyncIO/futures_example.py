import asyncio
import random


# Simulate an API call
async def api_call(endpoint):
    print(f"Calling API: {endpoint}")
    await asyncio.sleep(random.uniform(0.5, 2))  # Simulate network delay
    return {"data": f"Response from {endpoint}"}


# Worker function to fetch and process data
async def fetch_and_process_data(future, endpoint):
    try:
        # Step 1: Call the API
        raw_data = await api_call(endpoint)

        # Step 2: Process the result
        processed_data = {"processed": raw_data["data"].upper()}

        # Step 3: Store the processed data in the future
        future.set_result(processed_data)
    except Exception as e:
        future.set_exception(e)


# Consumer function to use the result
async def use_processed_data(future):
    print("Waiting for the processed data...")
    data = await future  # Wait until the future is resolved
    print("Got processed data:", data)


# Main entry point
async def main():
    loop = asyncio.get_running_loop()

    # Create a Future
    future = loop.create_future()

    # Start a worker to fetch and process data
    asyncio.create_task(fetch_and_process_data(future, "https://api.example.com/data"))

    # Use the data later in the program
    await use_processed_data(future)


asyncio.run(main())
