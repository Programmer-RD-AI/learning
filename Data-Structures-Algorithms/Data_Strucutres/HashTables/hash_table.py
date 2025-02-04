class Hash_Table:
    def __init__(self, size: int) -> None:
        self.data = [None] * size
        self.size = size

    def get_position(self, key):  # O(1)
        position = 0
        for i in range(len(key)):
            position += (
                ord(key[i]) * i
            )  # get the unicode character (between 0 to 65535) * index
        return position % self.size  # this get the position between 0 and the self.size
        # return hash(key) % self.size

    def set(self, key, value):  # O(1)
        position = self.get_position(key)
        if self.data[position] is None:
            self.data[position] = []
        self.data[position].append([key, value])
        return True

    def get(self, key):  # O(1) - O(n)
        position = self.data[self.get_position(key)]
        for pos in position:
            if pos[0] == key:
                return pos
        return None

    def keys(self):
        keys = []
        for partition in self.data:
            if partition is None:
                continue
            for key, val in partition:
                keys.append(key)
        return keys


myHashTable = Hash_Table(50)
print(myHashTable.set("grapes", 10000))
print(myHashTable.set("apples", 54))
print(myHashTable.set("orange", 2))
print(myHashTable.keys())
