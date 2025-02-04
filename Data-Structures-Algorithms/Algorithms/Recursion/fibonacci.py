def fibonacci(n):
    val = [0, 1]
    for i in range(2, n + 1):
        val.append(val[i - 1] + val[i - 2])
    return val.index(n)


print(fibonacci(5))


def fibonacci_r(n, array: list = [0, 1]):
    if array[-1] == n:
        return len(array) - 1
    array.append(array[len(array) - 1] + array[len(array) - 2])
    return fibonacci_r(n, array)


print(fibonacci_r(5))
