# NEW SYNTAX - New contextlib-based context manager

import contextlib


@contextlib.contextmanager
def retry():

    print("Entering Context")

    yield

    print("Exiting Context")
