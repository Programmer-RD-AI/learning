# Log all pairs of array

boxes = ["a", "b", "c", "d", "e"]  # [1, 2, 3, 4, 5]


def logAllPairsofArray(array):
    for i in range(len(array)):  # n
        for j in range(len(array)):  # n
            print(array[i], array[j])


logAllPairsofArray(boxes)
# O(n*n) -> O(n^2)
