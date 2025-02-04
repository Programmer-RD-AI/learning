numbers = [99, 44, 6, 2, 1, 5, 63, 87, 283, 4, 0]


def selectionSort(array: list) -> list:
    length = len(array)
    for idx in range(length):
        minimum = idx
        for i in range(minimum + 1, length):
            if array[i] < array[minimum]:
                minimum = i
        array[minimum], array[idx] = array[idx], array[minimum]
    return array


print(selectionSort(numbers))
