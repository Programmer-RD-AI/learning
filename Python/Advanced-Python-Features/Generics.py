class KVStore[K: str | int, V]:

    def __init__(self) -> None:

        self.store: dict[K, V] = {}

    def get(self, key: K) -> V:

        return self.store[key]

    def set(self, key: K, value: V) -> None:

        self.store[key] = value


kv = KVStore[str, int]()

kv.set("one", 1)

kv.set("two", 2)

kv.set("three", 3)


class Foo[UnBounded, Bounded: int, Constrained: int | float]:

    def __init__(self, x: UnBounded, y: Bounded, z: Constrained) -> None:

        self.x = x

        self.y = y

        self.z = z


class Tuple[*Ts]:

    def __init__(self, *args: *Ts) -> None:

        self.values = args


# Works with any number of types!

pair = Tuple[str, int]("hello", 42)

triple = Tuple[str, int, bool]("world", 100, True)
