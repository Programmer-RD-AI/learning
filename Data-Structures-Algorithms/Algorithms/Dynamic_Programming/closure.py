def make_counter():
    count = 0

    def inner():
        nonlocal count
        count += 1
        return count

    return inner


counter = make_counter()

c = counter()
print(c)

c = counter()
print(c)

c = counter()
print(c)
