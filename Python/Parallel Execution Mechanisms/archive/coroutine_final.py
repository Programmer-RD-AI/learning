import asyncio


async def fetch_data(delay, id):
    print("Fetching data... ", id)
    await asyncio.sleep(delay)
    print("Data fetched ", id)
    return {"data" + str(id): delay}


# coroutine function
async def main():
    print("Start of main coroutine")
    task = await fetch_data(1, 1)
    task = await fetch_data(
        2, 2
    )  # Here the this sub-routine is kept on hold and the other `fetch_data` one is started on working...
    task = await fetch_data(3, 3)
    # Only when await is called to a async function does the function get actually called
    print(task)


# main() -> Coroutine Object -> Needs to be awaited
# asyncio.run() -> Event Loop

asyncio.run(main())
