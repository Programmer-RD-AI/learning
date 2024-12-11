import asyncio


async def fetch_data(delay):
    print("Fetching data...")
    await asyncio.sleep(delay)
    print("Data fetched")
    return {"data": delay}


# coroutine function
async def main():
    print("Start of main coroutine")
    task = await fetch_data(
        2
    )  # Here the this sub-routine is kept on hold and the other `fetch_data` one is started on working...
    # Only when await is called to a async function does the function get actually called
    print(task)


# main() -> Coroutine Object -> Needs to be awaited
# asyncio.run() -> Event Loop

asyncio.run(main())
