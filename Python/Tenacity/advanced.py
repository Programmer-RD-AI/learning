from tenacity import retry, stop_after_delay, wait_exponential, retry_if_exception_type


@retry(
    stop=stop_after_delay(10),  # Stop after 10 seconds
    wait=wait_exponential(multiplier=1, min=2, max=5),  # Exponential backoff
    retry=retry_if_exception_type(
        (ValueError, KeyError)
    ),  # Retry on specific exceptions
)
def flaky_function():
    print("Trying...")
    raise ValueError("Transient error!")


try:
    flaky_function()
except ValueError as e:
    print(f"Operation failed: {e}")
