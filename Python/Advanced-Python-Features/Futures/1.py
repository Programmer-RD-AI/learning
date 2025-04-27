import asyncio


async def main():

    future = asyncio.Future()

    # Set result after delay

    asyncio.create_task(set_after_delay(future))

    # Await just like a JS Promise!

    result = await future

    print(result)  # "Worth the wait!"


async def set_after_delay(future):

    await asyncio.sleep(1)

    future.set_result("Worth the wait!")


asyncio.run(main())
