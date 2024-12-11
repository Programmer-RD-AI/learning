# Global Interpreter Lock (GIL)

A lock that allows only one thread to control the Python interpreter at once.

In turn only one thread can be in execution at any point of time. This is a performance bottleneck on multi threaded applications.

## What Problem Did the GIL Solve for Python?
