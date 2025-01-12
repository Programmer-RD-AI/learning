from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def unreliable_function():
    print("Attempting...")
    raise ValueError("Something went wrong!")


# Call the function
try:
    unreliable_function()
except ValueError as e:
    print(f"Failed after retries: {e}")
