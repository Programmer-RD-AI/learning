import asyncio


async def fetch_data(id, sleep_time):
    print(f"Coroutine {id} starting to fetch data.")
    await asyncio.sleep(sleep_time)
    return {"id": id, "sleep_time": sleep_time}


async def main():
    tasks = []
    # if any of the tasks fail it will stop the execution of the other tasks as well
    async with asyncio.TaskGroup() as tg:  # async context manager
        for i in range(1, 4):
            tasks.append(tg.create_task(fetch_data(i, i)))
    results = [task.result() for task in tasks]
    for result in results:
        print(result)
