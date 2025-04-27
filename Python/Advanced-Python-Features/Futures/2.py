from concurrent.futures import ThreadPoolExecutor

import time


def slow_task():

    time.sleep(1)

    return "Done!"


with ThreadPoolExecutor() as executor:

    # Returns a Future immediately

    future = executor.submit(slow_task)

    # Do other work while waiting...

    print("Working...")

    # Get result when needed

    print(future.result())
