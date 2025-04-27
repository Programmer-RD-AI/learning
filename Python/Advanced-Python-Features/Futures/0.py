from concurrent.futures import Future

import time, threading


# Create and manage a future manually

future = Future()


# Background task function


def background_task():

    time.sleep(2)

    future.set_result("Done!")


thread = threading.Thread(target=background_task)

thread.daemon = True

thread.start()


# Try all control operations

print(f"Cancelled: {future.cancel()}")  # Likely False if started


try:

    # Wait at most 0.5 seconds

    result = future.result(timeout=0.5)

except TimeoutError:

    print("Timed out!")


# Create failed future

err_future = Future()

err_future.set_exception(ValueError("Failed"))

print(f"Has error: {bool(err_future.exception())}")
