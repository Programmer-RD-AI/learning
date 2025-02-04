# Given 2 arrays,
# create a function that lets a user know true / false
# whether these two arrays contain any common items


def common(array1, array2):
    common_var = False
    for i in range(len(array1)):
        if array1[i] in array2:
            common_var = True
            break
    return common_var


array1 = ["a", "b", "c", "x"]
array2 = ["z", "y", "a"]
print(common(array1, array2))
