# Global Interpreter Lock (GIL)

A lock that allows only one thread to control the Python interpreter at once, which disallows parraleism.

In turn only one thread can be in execution at any point of time. This is a performance bottleneck on multi threaded applications.

## What Problem Did the GIL Solve for Python?

### Reference Counting for Memory Management

This keeps track of the number of references that point to the object.

When this count reaches zero, the occupied memory is released.

Example:

```python
>>> import sys
>>> a = []
>>> b = a
>>> sys.getrefcount(a)
3
```

In the above example, the reference count for the empty list object [] was 3. The list object was referenced by a, b and the argument passed to sys.getrefcount().

#### Connection to GIL

The problem that was there was that the reference count variable needed protection from conditions such as where 2 threads increase and decrease value simultaneously.
If this happens it can cause either a leaked memory never released or incorrectly release the memory while a reference to that object still exists.

The reference count variable can be kept safe by adding locks to all data structures that are shared across threads so that they are not modified simultaneously.

But adding the lock would means multile locks would exist which raise another problem.

- Deadlocks: Can only happen when there is more than 1 lock, and the decreased performance caused by repeatedly creating and releasing locks.

So with those into consideration the GIL a signel lock on the interpreter itself was introduced as the solution, which adds a rule that exueciton of any python bytecode requires getting the interpreter lock.
This removes the issue od deadlocks and doesnt introduce much performance overhead, while making any CPU bound Python program single threaded.

GIL is also used in languages such as Ruby, but it is not the only solution, some langauges avoid this using thread safe memory managemnt using approached such as garbage collection.

To compensate for the loss of the single threaded performance, JIT compilers are utilize for other performance boosting.

## Why Was the GIL Chosen as the Solution?

~"GIL is one of the things that made Python as popular as it is today." - Larry Hastings

Python has been around since the days without threads, and it was designed to be easy to use, to make development quicker.

A lot of extensions in python are written from the existing C libraries, and these C libraries required thread safe memory managerment which GIL provided. C lbiraries that were not thread safe became easier to integrater.

CPython developers faced these issues in Python' early life. (CPython was the most popular variation of Python back in the day)

## The Impact on Multi-Threaded Python Programs

### CPU Bound Programs

Limited by the processing power of the CPU, such as computations where the CPU busy, and since the CPU is busy at that time, these programs dont benefit from the multi threading because of the GIL that prevents true parallel execution.

```python
import threading

def compute():
    # Simulate heavy computation
    total = 0
    for i in range(10**7):
        total += i
    print(total)

# Creating multiple threads for a CPU-bound task
threads = [threading.Thread(target=compute) for _ in range(4)]

for t in threads:
    t.start()

for t in threads:
    t.join()
```

### I/O Bound Programs

Tasks limited by the speed of Input Output Operations, such as disk reads/writes or network communications. Where the CPU is idle, in these scenarios multi threaded or asynchronous programming improve performance a lot. GIL is released while I/O operations are occurring allowing the other threads to run in the meantime.

```python
import threading
import time

def read_file():
    time.sleep(2)  # Simulate I/O delay
    print("File read complete")

# Creating multiple threads for an I/O-bound task
threads = [threading.Thread(target=read_file) for _ in range(4)]

for t in threads:
    t.start()

for t in threads:
    t.join()
```

## Why Hasn’t the GIL Been Removed Yet?

The possibility has been brought forth multiple times, but it would casuae a lot of backward incompatibility issues.
The GIL can be removed and has been removed multiple times in the past by developers and researchers, but all those broke the existing C extentions which depend on the GIL implementaiton heavily.
The creator of Python (Guido Van Rossum) has told the community on his article "It isn't easy to remove the GIL"

> “I’d welcome a set of patches into Py3k only if the performance for a single-threaded program (and for a multi-threaded but I/O-bound program) does not decrease”

And this condition hasn’t been fulfilled by any of the attempts made since.

## References

https://realpython.com/python-gil/#why-was-the-gil-chosen-as-the-solution
https://youtu.be/XVcRQ6T9RHo?si=TP7e1bqsx4OH37Ep
