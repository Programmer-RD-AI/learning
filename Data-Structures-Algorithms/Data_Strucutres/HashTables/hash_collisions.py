def scream_function():
    print("ahhh!")


user = {"age": 54, "name": "Kylie", "magin": True, "scream": scream_function}
user["age"]  # O(1)
user["spell"] = "abra kadabra"  # O(1)
user["scream"]()  # O(1)
print(hash("test"))
print(set({"a": 1, "b": 3}))
