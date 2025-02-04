# Heap Introduction

There is also smthn called: Heap that is used for Storage for garbage collection but this heap is the data structure used to manage information, also called binary heap

Features:
All levels are filled except the lowest
Lowest left is filled starting from the lext

Uses:
heapsort
Priority Queues

Types:
max-heap
min-heap

Height - O(log n)

## Max Heap

Value if the node i <= value of the parent
Used for heap sort

## Min Heap

Value of i >= value of the parent
Priority Queues

## In array form

Starts from 1
[21,17,15,14,9,12,13,8,5,1]
left(i) = 2 *i
right(i) = 2* i + 1
parent(i) = floor(i/2)

# Heap Methods

## max-heapify

Maintain max heap property is that the value of i <= value of parent
OR value of i >= value of children

Inputs: array, heap size (length of the heap), index

## build-max-heap

Wrapper function that calls max-heapify

# Get Leaves

A - the array
n - the number of leaves

leaves = A[floor(n/2) + 1] to A[n]
