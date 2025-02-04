# Parallel Execution Mechanisms

## Concurrency

## AsyncIO

For magnaging many waiting tasks.
A simple way to look at it is that we make sure our single threaded application is doing something always, ensuring that if something idle or waiting on something that while it is waiting something else is completing...

### Event Loop

The core that manages and distributes tasks.

### CoRoutines

Different functions that can be executed.

### Tasks

Run multiple co routines at the same time. (Simultaneously)

## Threading

For parallel tasks that share data with minimal CPU use

## Processing

For maximizing perofmrance on cpu inteisve tasks
