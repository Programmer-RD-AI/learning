class Graph:
    def __init__(self, initial_graph: dict = {}) -> None:
        self.adjacentList = initial_graph

    def addVertex(self, node: int) -> None:
        if node not in self.adjacentList.keys():
            self.adjacentList[node] = []
        return None

    def addEdge(self, node1: int, node2: int):
        if node2 not in self.adjacentList[node1]:
            self.adjacentList[node1].append(node2)
        if node1 not in self.adjacentList[node2]:
            self.adjacentList[node2].append(node1)
        return True

    def showConnections(self):
        keys = self.adjacentList.keys()
        for key in keys:
            print(f"{key} -> {self.adjacentList[key]}")


g = Graph()
g.addVertex("0")
g.addVertex("1")
g.addVertex("2")
g.addVertex("3")
g.addVertex("4")
g.addVertex("5")
g.addVertex("6")
g.addEdge("3", "1")
g.addEdge("3", "4")
g.addEdge("4", "2")
g.addEdge("4", "5")
g.addEdge("1", "2")
g.addEdge("1", "0")
g.addEdge("0", "2")
g.addEdge("6", "5")
g.showConnections()
