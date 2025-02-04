def repeat(array: list) -> int:
    exist = set()
    for item in array:
        if item not in exist:
            exist.add(item)
        else:
            return item
    return None


array1 = [2, 5, 1, 2, 3, 5, 1, 2, 4]
array2 = [2, 1, 1, 2, 3, 5, 1, 2, 4]
array3 = [2, 3, 4, 5]
print(repeat(array1))
print(repeat(array2))
print(repeat(array3))
