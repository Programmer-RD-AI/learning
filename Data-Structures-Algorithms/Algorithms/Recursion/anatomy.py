counter = 0

def inception():
    if counter > 3:
        return "done"
    counter += 1
    return inception()
