from typing import Literal, overload


@overload
def transform(data: str, mode: Literal["split"]) -> list[str]:
    print(mode, data)


@overload
def transform(data: str, mode: Literal["upper"]) -> str:
    print(data, mode)


def transform(data: str, mode: Literal["split", "upper"]) -> list[str] | str:

    if mode == "split":

        return data.split()

    else:

        return data.upper()


split_words = transform("hello world", "split")  # Return type is list[str]

split_words[0]  # Type checker is happy


upper_words = transform("hello world", "upper")  # Return type is str

upper_words.lower()  # Type checker is happy


upper_words.append("!")  # Cannot access attribute "append" for "str"
