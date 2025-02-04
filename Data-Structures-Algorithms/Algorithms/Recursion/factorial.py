tot = 1


def factorial(num: int, tot) -> int:
    if num != 0:
        tot *= num
        return factorial(num - 1, tot)
    return tot


tot = factorial(5, tot)
print(tot)
