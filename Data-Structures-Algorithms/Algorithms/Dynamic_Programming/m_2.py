import time

# its better for this cache to be inside the function because it is only used inside it and currently it is just polluting the global variable space


def memoizedAddTo80():
    cache = {}

    def inner(n):
        nonlocal cache
        if n in cache:
            return cache[n]
        cache[n] = n + 80
        return cache[n]

    return inner


memoized = memoizedAddTo80()
start = time.time()
print(memoized(5))
print(
    memoized(5)
)  # if i do this again i would have to go over the process again, and process the data again in turn it is not "efficient"
end = time.time()
print(end - start)
