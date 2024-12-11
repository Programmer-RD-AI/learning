import asyncio


async def set_future_result(future, value):
    await asyncio.sleep(2)
    future.set_result({value: value})


async def main():
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    asyncio.create_task(set_future_result(future, "Future result is ready"))

    print(await future)


asyncio.run(main())

