import asyncio


async def random_calculation():
    print("Random Calculation Started")
    await asyncio.sleep(5)  # Simulates a long-running operation
    print("Random Calculation Ended")
    return 50 * 50 * 50


async def some_other_task():
    print("Some other task started")
    await asyncio.sleep(10)  # Simulates a shorter operation
    print("Some other task ended")


async def main():
    print("Main task started")

    # Start the random_calculation task concurrently
    calculation_task = asyncio.create_task(random_calculation())

    # Perform some other work concurrently
    await some_other_task()
    print("await some_other_task")
    # Wait for the random_calculation to finish
    # value = await calculation_task
    print(f"Random Calculation Result:")

    print("Main task finished")


asyncio.run(main())
