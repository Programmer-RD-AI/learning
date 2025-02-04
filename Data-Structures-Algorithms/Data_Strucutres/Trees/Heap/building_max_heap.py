def max_heapify(a, heap_size, i):
    l = 2 * i
    r = 2 * i + 1
    largest = i
    if l < heap_size and a[l] > a[i]:
        largest = l
    if r < heap_size and a[r] > a[i]:
        largest = r
    if largest != i:
        a[largest], a[i] = a[i], a[largest]
        largest, a, heap_size, i = max_heapify(a, heap_size, largest)
    return largest, a, heap_size, i


def build_max_heap(a):
    heap_size = len(a)
    for i in range(heap_size // 2, 0, -1):
        largest, a, heap_size, i = max_heapify(a, heap_size, i)




print(lex("fsg"))
