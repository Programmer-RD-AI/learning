# Edge List
### Shows the connection between nodes
graph = [[0, 2], [2, 3], [2, 1], [1, 3]]

# Adjacent List
graph = [[2], [2, 3], [0, 1, 3], [1, 2]]

# Adjacent Matrix
### 1 is where the connection is established, with the value of the node taken into consideraion as the index of the list
graph = [[0, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]
graph = {0: [0, 0, 1, 0], 1: [0, 0, 1, 1], 2: [1, 1, 0, 1], 3: [0, 1, 1, 0]}
