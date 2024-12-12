# Global Interpreter Lock (GIL)

A lock that allows only one thread to control the Python interpreter at once.

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

