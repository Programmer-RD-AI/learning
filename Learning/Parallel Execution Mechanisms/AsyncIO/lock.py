import asyncio

share_resource = 0

lock = asyncio.Lock()


async def modify_shared_resource():
    global share_resource
    async with lock:
        print(f"Resource before modification: {share_resource}")
        share_resource += 1
        await asyncio.sleep(1)
        print(f"Resource after modification: {share_resource}")


async def main():
    await asyncio.gather(*(modify_shared_resource() for _ in range(5)))


asyncio.run(main())
