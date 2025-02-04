class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


class Stacks:
    def __init__(self):
        self.head = None

    def push(self, value: any) -> Node:
        if self.head:
            new_node = Node(value)
            new_node.next = self.head
            self.head = new_node
        else:
            self.head = Node(value)
        return self.head

    def pop(self) -> Node or None:
        if self.head:
            previous_head = self.head
            self.head = self.head.next
            return previous_head
        return self.head

    def peek(self) -> Node or None:
        if self.head:
            return self.head.data
        return self.head

    def lookup(self, value) -> (bool, Node):
        iterator = self.head
        while iterator:
            if iterator.data == value:
                return True, iterator
            iterator = iterator.next
        return False, Node(None)


s = Stacks()
for i in range(1000):
    s.push(f"sample{i}")
print(s.lookup("sample69"))
