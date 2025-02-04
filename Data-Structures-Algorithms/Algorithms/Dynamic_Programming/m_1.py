import time


def addTo80(n):
    return n + 80


start = time.time()
print(addTo80(5))
print(
    addTo80(5)
)  # if i do this again i would have to go over the process again, and process the data again in turn it is not "efficient"
end = time.time()
print(end - start)
cache = {}


def memoizedAddTo80(n):
    if n in cache:
        return cache[n]
    cache[n] = n + 80
    return cache[n]


start = time.time()
print(memoizedAddTo80(5))
print(
    memoizedAddTo80(5)
)  # if i do this again i would have to go over the process again, and process the data again in turn it is not "efficient"
end = time.time()
print(end - start)
