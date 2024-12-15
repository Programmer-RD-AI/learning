# Threading

Threading is where multiple threads are utilize to simulatniously run tasks.
So for example in the `main.py` there are `time.sleep()` times and the highest one is the time that the time would take since the others will run until its not yk? so it will go back and for the most active thread, if its not running then it will be running in the background.

You can create a thread:
```python

import threading

# FUNC_NAME -> The Function passed such as `walk_my_dog` without calling the function it self
# ARGUMENTS -> A tuple of arguments that will be passed to the function

threading.Thread(FUNC_NAME, args=ARGUMENTS).start()

```

You can run a dameon thread (it is pretty much a background thread that doesnt have any sort of priority so if the main non dameon thread's finish the dameon thread dies no matter if its done or not):
```python

import threading

threading.Thread(FUNC_NAME, args=ARGUMENTS, dameon=True).start()
```

Make sure the Thread finishes:
```python

import threading

t1 = threading.Thread(FUNC_NAME, args=ARGUMENTS)
t1.start()
t1.join() # Make's sure the thread finishes

```
