class Node:
    def __init__(self, data: any, next=None) -> None:
        self.data = data
        self.next = next


class Queues:
    def __init__(self) -> None:
        self.head = None

    def enqueue(self, data) -> Node:
        if self.head:
            new_node = Node(data, next=self.data)
            self.head = new_node
        else:
            self.head = Node(data)
        return self.head

    def dequeue(self) -> Node:
        if self.head:
            self.head = self.data.next
        return self.head

    def peek(self) -> any:
        return self.head.data

    def lookup(self, data) -> (bool, Node):
        iteration = self.head
        while iteration:
            if iteration.data == data:
                return (True, data)
            iteration = iteration.next
        return (False, Node(data))
