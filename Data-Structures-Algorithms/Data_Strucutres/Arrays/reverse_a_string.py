# Create a function that reveres a string
# Hi my name is Andrei should become:
# ierdnA si eman iH


def reverse(strings: str) -> str:
    final_string = ""
    strings = list(strings)
    length = len(strings)
    for _ in range(length):
        final_string += strings[length-1]
        length -= 1
    return final_string


print(reverse("Hi my name is Andrei"))
