class Node:
    def __init__(self, data):
        self.data = data
        self.right = None
        self.left = None


class Trie:
    def __init__(self):
        self.root = Node(None)
        self.inserted = []

    def createNodes(self, remaining_text: str):
        if remaining_text == "":
            return None
        remaining_text = list(remaining_text)
        inital = Node(remaining_text[0])
        prev = inital
        remaining_text.pop(0)
        while remaining_text != []:
            new_node = Node(remaining_text[0])
            prev.left = new_node
            remaining_text.pop(0)
            prev = new_node
        return inital

    def checkLastPoint(self, node: Node, value):
        print(
            value,
            node.right,
        )
        if value == "":
            return node, value
        first_index = value[0] if len(value) >= 1 else ""
        if node.right:
            print(node.right.data == first_index, node.right.data, first_index)
            if node.right.data == first_index:
                return self.checkLastPoint(node.right, value[1:])
        if node.left:
            print(node.left.data == first_index, node.left.data, first_index)
            if node.left.data == first_index:
                return self.checkLastPoint(node.left, value[1:])
        print(value)
        return node, value

    def insert(self, word: str) -> None:
        if word in self.inserted:
            return None
        word = word.lower()
        self.inserted.append(word)
        iterator = self.root
        if iterator.right:
            iterator, word = self.checkLastPoint(iterator.right, word)
        if iterator.left:
            iterator, word = self.checkLastPoint(iterator.left, word)
        print(word, iterator, iterator.left, iterator.right)
        rest_of_the_node = self.createNodes(word)
        if iterator.right:
            iterator.right = rest_of_the_node
        else:
            iterator.left = rest_of_the_node
        return rest_of_the_node

    def print(self, root=None):
        iterator = root if root else self.root
        if iterator.left:
            print(iterator.left.data, iterator.left)
            self.print(iterator.left)
        if iterator.right:
            print(iterator.right.data, iterator.right)
            self.print(iterator.right)

    def search(self, word: str) -> bool:
        if word in self.inserted:
            return True
        word = word.lower()
        node, remaining_value = self.checkLastPoint(self.root, word)
        if node.right or node.left:
            return False
        return True if remaining_value == "" else False

    def startsWith(self, prefix: str) -> bool:
        print("\n")
        self.print()
        prefix = prefix.lower()
        _, remaining_value = self.checkLastPoint(self.root, prefix)
        return True if remaining_value == "" else False
