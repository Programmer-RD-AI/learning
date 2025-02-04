import time


def fetch_data(url):
    print(f"Fetching data from {url}...")
    time.sleep(2)  # Simulate a delay
    print(f"Finished fetching data from {url}")


def synchronous_main():
    urls = ["URL1", "URL2", "URL3"]
    for url in urls:
        fetch_data(url)


print("Synchronous Start")
start_time = time.time()
synchronous_main()
print(f"Synchronous End (Duration: {time.time() - start_time:.2f} seconds)")
