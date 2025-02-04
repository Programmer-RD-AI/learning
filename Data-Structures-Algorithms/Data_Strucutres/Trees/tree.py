class Node:
    def __init__(self, value: int, right=None, left=None) -> None:
        self.value = value
        self.right = right
        self.left = left
        self.copies = 0


class Tree:
    def __init__(self, root: int) -> None:
        self.data = Node(root)
        self.length = []

    def insert(self, value: int) -> None:
        element = Node(value)
        iterator = self.data
        while iterator:
            if value == iterator.value:
                iterator.copies += 1
                element = iterator
            if value > iterator.value and iterator.right is None:
                iterator.right = element
                break
            if value < iterator.value and iterator.left is None:
                iterator.left = element
                break
            iterator = iterator.right if value > iterator.value else iterator.left
        return element

    def lookup(self, value: int) -> None:
        iterator = self.data
        while iterator:
            if value == iterator.value:
                return iterator
            iterator = self.check_difference(value, iterator)
        return None

    def check_difference(
        self, value: int, iterator: Node, string_out: bool = False
    ) -> Node:
        if value > iterator.value:
            iterator = iterator.right if not string_out else "right"
        else:
            iterator = iterator.left if not string_out else "left"
        return iterator

    def remove(self, value: int) -> (Node, Node):
        iterator = self.data
        prev = self.data
        while iterator:
            iterator = self.check_difference(value, iterator)
            if not iterator:
                break
            if value == iterator.value:
                side = self.check_difference(value, prev, string_out=True)
                if iterator.right is None and iterator.left is None:
                    if side == "right":
                        prev.right = None
                    else:
                        prev.left = None
                elif iterator.right is None:
                    if side == "right":
                        prev.right = iterator.left
                    else:
                        prev.left = iterator.left
                elif iterator.left is None:
                    if side == "right":
                        prev.right = iterator.right
                    else:
                        prev.left = iterator.right
                else:
                    lowest = (
                        iterator.left
                        if iterator.left.value < iterator.right.value
                        else iterator.right
                    )
                    highest = (
                        iterator.left
                        if iterator.left.value > iterator.right.value
                        else iterator.right
                    )
                    lowest.right = highest
                    if side == "right":
                        prev.right = lowest
                    else:
                        prev.left = lowest
                break
            prev = iterator
        return prev, iterator


t = Tree(69)
t.insert(96)
t.insert(9855)
t.insert(95)
t.insert(50)
found = t.lookup(96)
prev, iterator = t.remove(96)
print(prev.value, iterator)
found = t.lookup(96)
print(found)
found = t.lookup(95)
print(found)
print(t.lookup(69).right.right.value)
