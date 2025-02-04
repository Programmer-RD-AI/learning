from collections import deque


class Node:
    def __init__(self, value) -> None:
        self.value = value
        self.right = None
        self.left = None


class BinarySearchTree:
    def __init__(self, root=None):
        self.root = root

    def insert(self, value):
        n = Node(value)
        if not self.root:
            self.root = n
            return n
        iterator = self.root
        while iterator:
            og_iterator = iterator
            if iterator.right and iterator.value < value:
                iterator = iterator.right
            if iterator.left and iterator.value > value:
                iterator = iterator.left
            if iterator == og_iterator:
                if iterator.value > value:
                    iterator.left = n
                else:
                    iterator.right = n
                return n
        return None

    def lookup(self, value) -> list or bool:
        prev = None
        iterator = self.root
        while iterator:
            if iterator.value == value:
                return [iterator, prev]
            prev = iterator
            if iterator.right and iterator.value < value:
                iterator = iterator.right
            if iterator.left and iterator.value > value:
                iterator = iterator.left
            if iterator == prev:
                break
        return None

    def remove(self, value):
        existence = self.lookup(value)
        if not existence:
            return False
        iterator, prev = existence
        switch = None
        if iterator.left:
            switch = iterator.left
        elif iterator.right:
            switch = iterator.right
        if iterator.value > prev.value:
            prev.right = switch
        else:
            prev.left = switch

    def print(self, root=None):
        iterator = root if root else self.root
        print(iterator.value)
        if iterator.right:
            self.print(iterator.right)
        if iterator.left:
            self.print(iterator.left)

    def bfs(self):
        queue = deque()
        visited = []
        queue.append(self.root)
        while len(queue) > 0:
            currentNode = queue.pop()
            if currentNode.right:
                queue.append(currentNode.right)
            if currentNode.left:
                queue.append(currentNode.left)
            visited.append(currentNode.value)
        return visited, queue

    def dfs_in_order(self):
        return traverseInOrder(self.root, [])

    def dfs_post_order(self):
        return traversePostOrder(self.root, [])

    def dfs_pre_order(self):
        return traversePreOrder(self.root, [])


def traverseInOrder(node, data):
    if node.left:
        traverseInOrder(node.left, data)
    data.append(node.value)
    if node.right:
        traverseInOrder(node.right, data)
    return data


def traversePreOrder(node, data):
    data.append(node.value)
    if node.left:
        traversePreOrder(node.left, data)
    if node.right:
        traversePreOrder(node.right, data)
    return data


def traversePostOrder(node, data):
    if node.left:
        traversePostOrder(node.left, data)
    if node.right:
        traversePostOrder(node.right, data)
    data.append(node.value)
    return data


tree = BinarySearchTree()
tree.insert(9)
tree.insert(4)
tree.insert(6)
tree.insert(20)
tree.insert(170)
tree.print()
tree.remove(170)
tree.print()
print(tree.bfs())
