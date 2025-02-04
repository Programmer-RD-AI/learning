# Big O: "How Code Slows as data grows"

1. Describes the performance of an algorithm as the amount of data increases
2. Machine Independent (# steps to completion)
3. Ignore Smaller Operations O(n+1) -> O(n)

Example:
- O(1)
  - Constant Time
  - Random access of an element in a array
  - Insering at the beginning of a linkedlist
- O(n)
  - Linear Time
  - Looping through elements in an array
  - Searching through a linkedlist
- O(log n)
  - Logarithmic time
  - Binary search
- O(n log n)
  - Quasilinear time
  - Quick sort
  - Merge sort
  - Heap Sort
- O(n^2)
  - Quadartic time
  - Insertion sort
  - Selection sort
  - bubble sort
- O(n!)
  - Factorial Time
  - Traveling Salesman Problem

Note: n = amount of data (its a variable like X)

## O(n) linear time

```java
int addUp(int n){
    int sum = 0;
    for (int i = 0; i <= n; i++){
        sum += i;
    }
    return sum;
}
```

n = 100000

~100000 steps

## O(1) constant time

```java
int addUp(int n){
    int sum = n * (n + 1) / 2;
    return sum;
}
```

n = 10000

~3 steps

