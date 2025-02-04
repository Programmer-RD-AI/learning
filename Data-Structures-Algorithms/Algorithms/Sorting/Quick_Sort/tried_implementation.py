final_array = []


def bigger(array, value):
    index = array.index(value)
    for element in array[:index]:
        if element > value:
            return True
    return False


def quickSort(array, final_array):
    if len(array) == 2:
        final_array.extend(array)
        return final_array
    print(array)
    pivot = array[-1]
    while bigger(array, pivot):
        for element in array:
            print(element, element > pivot, array)
            if element > pivot:
                array.remove(element)
                array.append(element)
    split_1 = array[pivot:]
    split_2 = array[: pivot - 1]
    final_array.append(pivot)
    if split_1 != []:
        return quickSort(split_1, final_array)
    if split_2 != []:
        return quickSort(split_2, final_array)
    return final_array


print(quickSort([3, 7, 8, 5, 2, 1, 9, 5, 4], final_array))
