numbers = [99, 44, 6, 2, 1, 5, 63, 87, 283, 4, 0]
# numbers = [2, 1, 6, 5]


def mergeSort(array: list):
    if len(array) == 1:
        return array
    middle = int(len(array) / 2)
    left = array[middle:]
    right = array[:middle]
    return merge(left, right)


def merge(left: list, right: list):
    left_element = mergeSort(left)
    right_element = mergeSort(right)
    if left_element[0] < right_element[0]:
        left_element.extend(right_element)
        return left_element
    right_element.extend(left_element)
    return right_element


print(mergeSort(numbers))
