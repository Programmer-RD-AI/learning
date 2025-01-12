import asyncio


async def access_resource(semaphore, resource_id):
    async with semaphore:
        print("Accessing resource")
        await asyncio.sleep(1)
        print(f"Releasing {resource_id} accessed")


async def main():
    semaphore = asyncio.Semaphore(2)  # Allow only to be running at the same time twice
    await asyncio.gather(*(access_resource(semaphore, i) for i in range(4)))


asyncio.run(main())
