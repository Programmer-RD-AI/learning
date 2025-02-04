def findPosition(array: list, value: int):
    for i in range(len(array) - 1):
        if array[i] < value and array[i + 1] > value:
            return i + 1
    return len(array) if array and array[-1] < value else 0


def insertionSort(array: list) -> list:
    sorted_array = []
    for element in array:
        position = findPosition(sorted_array, element)
        sorted_array.insert(position, element)
    return sorted_array


print(insertionSort([99, 44, 6, 2, 1, 5, 63, 87, 283, 4, 0]))
