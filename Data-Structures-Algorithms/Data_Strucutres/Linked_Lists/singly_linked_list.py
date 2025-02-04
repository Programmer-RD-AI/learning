# myLinkedList = {
#     "head": {"value": 10, "next": {"value": 5, "next": {"value": 16, "next": None}}}
# }


class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


class LinkedList:
    def __init__(self, value: any) -> None:
        self.head = Node(value)
        self.tail = self.head
        self.length = 1

    def append(self, value: any) -> Node:
        new_info = Node(value)
        self.tail.next = new_info
        self.tail = new_info
        self.length += 1
        return new_info

    def prepend(self, value: any) -> Node:
        new_info = Node(value, self.head)
        self.head = new_info
        self.length += 1
        return new_info

    def insert(self, position: int, value: any):
        if position <= 0:
            return self.prepend(value)
        elif position >= self.length:
            return self.append(value)
        iterator = self.head
        count = 1
        while iterator:
            if count == position:
                new_node = Node(value, next=iterator.next)
                iterator.next = new_node
                break
            iterator = iterator.next
            count += 1
        return new_node

    def remove(self, value):
        iterator = self.head
        prev = self.head
        while iterator:
            if iterator.data == value:
                prev.next = iterator.next
                self.length -= 1
                break
            prev = iterator
            iterator = iterator.next
        return self.head

    def print_out(self):
        iterator = self.head
        while iterator:
            print(iterator.data, iterator.next)
            iterator = iterator.next

    def reverse(self):
        prev = None
        iterator = self.head
        while iterator:
            next = iterator.next
            iterator.next = prev
            prev = iterator
            iterator = next
        self.head = prev
        return self.head


myLinkedList = LinkedList(10)
myLinkedList.append(5)
myLinkedList.append(16)
myLinkedList.prepend(1)
myLinkedList.insert(2, 69)
myLinkedList.remove(5)
myLinkedList.print_out()
print("\n")
myLinkedList.reverse()
myLinkedList.print_out()
