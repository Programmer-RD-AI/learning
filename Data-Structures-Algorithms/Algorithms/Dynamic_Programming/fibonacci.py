import time


def fibonacci():
    fibonacci = [0, 1]

    def inner(n):
        nonlocal fibonacci
        tot = 0
        for i in range(1, n):
            if i + 1 < len(fibonacci):
                tot += fibonacci[i]
            else:
                iter_tot = fibonacci[i - 2] + fibonacci[i - 1]
                fibonacci.insert(i, iter_tot)
                tot += iter_tot
        return tot

    return inner


start = time.time()
fib = fibonacci()
print(fib(10))
end = time.time()
print(end - start)
start = time.time()
print(fib(10))
end = time.time()
print(end - start)
