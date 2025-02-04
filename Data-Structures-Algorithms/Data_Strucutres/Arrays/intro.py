strings = ["a", "b", "c", "d"]
# if we were in a 32 bit mode then it would be 4*4 = 16bytes

print(strings[2])

# push
strings.append("e")  # O(1)

print(strings)

# remove
strings.remove(strings[-1])  # O(1)
print(strings)

# add to start
strings.insert(0, "s")  # O(n)
print(strings)

# add to middle
strings.insert(round(len(strings) / 2), "alien")  # O(n/2)
print(strings)
