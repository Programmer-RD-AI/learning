import time


def fibonacci():
    cache = {}

    def inner(n):
        nonlocal cache
        if n in cache:
            return cache[n]
        else:
            if n < 2:
                return n
            f = fibonacci()
            cache[n] = f(n - 1) + f(n - 2)
            return cache[n]

    return inner


start = time.time()
f = fibonacci()
print(f(10))
end = time.time()
print(end - start)
