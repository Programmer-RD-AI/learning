## Effects of GIL

CPU -> Thread & Cores
One core can have 1 task in parallel
More cores the more tasks we can process at the same exact time.
The tasks that CPU cores handle are Software threads
So each one of these threads would be executed by one core one at a time in a single threaded application.
We can split up our task into multiple threads in which they can be exeucted in multiple CPU cores.
So where our threads would need to connect with the Interpreter to execute the code, they are usually blocked since 1 thread would have started off using it and the GIL has been placed.

### Software Threads Vs. Hardware Threads

Software threads are created by programs or applications, while hardware threads exist on the physical CPU itself.

### Process

Instance of a computer program that is being executed.
Basic Components:

- Executable Program
- Associated data needed by the program
- Execution context of the program

### Multi Processing

This is the easiest solution that is provided by python where each process would have its own interpreter in turn allowing for true parallelism
