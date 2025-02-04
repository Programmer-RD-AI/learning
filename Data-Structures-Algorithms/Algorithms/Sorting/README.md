# Introduction

In an interview it is really important. Sorting is not a big deal when it comes to small data, but when it comes to large data, we must focus on optimization where we need to learn about the types of sorting algorithms

Algorithms Covered:

- Bubble Sort
- Insertion Sort
- Selection Sort
- Merge Sort
- Quick Sort

# Issue with `sort()`

There are many ways we can sort things and with small differences there is a lot to be taken into consideration.

# Sorting Algorithms

- Radix Sort
- Quick Sort
- Heap Sort
- Bubble Sort
- Selection Sort
- Insertion Sort
- Merge Sort
- Counting Sort

[Visualize Sorting Algorithms](https://www.toptal.com/developers/sorting-algorithms)

# Stable vs Unstable Algorithms

A sorting algorithm is said to be stable if two objects with equal keys appear in the same order in sorted output as they appear in the input array to be sorted.
In an unstable sort algorithm, straw or spork may be interchanged, but in a stable one, they stay in the same relative positions

[More Info](https://stackoverflow.com/questions/1517793/what-is-stability-in-sorting-algorithms-and-why-is-it-important)

# Which Sort is Best?

## Insertion Sort

Few items only, or items are mostly sorted

## Bubble Sort

Extremly unlikely to be used in the real world

## Selection Sort

Same as bubble sort, and its wont be used ever, and it is used for teaching sorting

## Merge Sort

Really good due to divide and concur, Time complexity is always guaranteed to be O(n log(n)). Use if time complexity is the biggest concern. But for space complexity it isnt the best.

## Quick Sort

Better than quick sort..? Only one downside is the worst case of O(n^2). If we are worried about the worst case scenario then we would rather use merge sort

# Review

Most of the time when talking about sorting algorithms its mostly O(n log (n))
Quick Sort use when the average case performance matters more than the worst case performance
Merge sort on the other hand is great because it is always has the time complexity, but the space complexity is a little bit higher its worth it
Insertion, Selection, Bubble are useful to use when you are just using them in small pet or so projects, but in real life you will most probably be using the programming languages or framework
