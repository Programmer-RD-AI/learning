class Stacks:
    def __init__(self) -> None:
        self.data = []

    def pop(self) -> list:
        self.data.pop()
        return self.data

    def push(self, value: any) -> list:
        self.data.append(value)
        return self.data

    def lookup(self, value: any) -> (bool, any):
        for d in self.data:
            if d == value:
                return True, d
        return False, d

    def peek(self) -> any:
        return self.data[-1]
