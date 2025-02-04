numbers = [99, 44, 6, 2, 1, 5, 63, 87, 283, 4, 0]


def bubbleSort(array: list) -> list:
    iterations = 0
    while True:
        swap = False
        for idx in range(len(array) - 1):
            iterations += 1
            if array[idx] > array[idx + 1]:
                swap = True
                temp = array[idx]
                array[idx] = array[idx + 1]
                array[idx + 1] = temp
        if not swap:
            break
    print(iterations)
    return array


print(bubbleSort(numbers))


def bubbleSort(array: list) -> list:
    iterations = 0
    length = len(array)-1
    for _ in range(length):
        for i in range(length):
            iterations += 1
            if array[i] > array[i + 1]:
                temp = array[i]
                array[i] = array[i + 1]
                array[i + 1] = temp
    print(iterations)
    return array


print(bubbleSort(numbers))
