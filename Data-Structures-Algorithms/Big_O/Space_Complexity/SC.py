def boo(n):
    for i in range(len(n)):
        print("boo")


boo([1, 2, 3, 4, 5])  # O(1) # we arent effecting memeory at all


def arrayoOfHiNTTimes(n):
    hiArray = []
    for i in range(n):
        hiArray[i] = "hi"
    return hiArray


arrayoOfHiNTTimes(6)  # O(n) # we are effecting memory becze of hiArray[i]='hi'
