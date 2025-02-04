class Node:
    def __init__(self, data: any, next=None, previous=None) -> None:
        self.data = data
        self.next = next
        self.previous = previous


class Linked_List:
    def __init__(self, data: any) -> None:
        self.head = Node(data)
        self.tail = self.head
        self.length = 1

    def append(self, value: any) -> None:
        new_element = Node(value, previous=self.tail)
        self.tail.next = new_element
        self.tail = new_element
        self.length += 1
        return new_element

    def prepend(self, value: any) -> Node:
        new_element = Node(value, next=self.head)
        self.head.previous = new_element
        self.head = new_element
        self.length += 1
        return new_element

    def insert(self, key: any, value: any) -> Node:
        iterator = self.head
        count = 0
        while iterator:
            if count == key:
                new_element = Node(value, previous=iterator, next=iterator.next)
                iterator.next.previous = new_element
                iterator.next = new_element
            count += 1
            iterator = iterator.next
        return new_element

    def remove(self, value: any) -> bool:
        iterator = self.head
        prev = self.tail
        while iterator:
            if iterator.data == value:
                prev.next = iterator.next
                iterator.next.previous = prev
                self.length -= 1
                return True
            prev = iterator
            iterator = iterator.next
        return False

    def print_out(self):
        iterator = self.head
        while iterator:
            print(iterator.data, iterator.next, iterator.previous)
            iterator = iterator.next


myLinkedList = Linked_List(10)
myLinkedList.append(5)
myLinkedList.append(16)
myLinkedList.prepend(1)
myLinkedList.insert(2, 69)
myLinkedList.remove(5)
myLinkedList.print_out()
