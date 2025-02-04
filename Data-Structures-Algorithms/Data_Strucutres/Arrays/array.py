# a = []
class Array:
    def __init__(self, data: list = None) -> None:
        self.length = len(data) if data is not None else 0
        self.data = data if data is not None else {}

    def __getitem__(self, key) -> any:
        return self.data[key]

    def append(self, value: any) -> list:
        self.data[self.length] = value
        self.length += 1
        return self.data

    def delete(self, key: int) -> list:
        new_array = {}
        for i in range(self.length):
            if key != i:
                new_array[i] = self.data[i]
        self.data = new_array
        return new_array

    def update(self, key: int, value: any) -> list:
        self.data[key] = value
        return self.data


a = Array()
print(a.append("test"))
print(a.append("test"))
print(a.append("test"))
print(a.append("test"))
print(a.update(2, "again"))
