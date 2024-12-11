import asyncio


async def waiter(event):
    await event.wait()


async def setter(event):
    await event.set()


async def main():
    event = asyncio.Event()
    await asyncio.gather(setter(event), waiter(event))

asyncio.run(main())
