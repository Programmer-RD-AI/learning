# Linked Lists

## Introduction

Singly and Doubly Linked Lists
Problems with Arrays: When scaling the size of an array, it is allocated somewhere else and in turn there is a performance problem there, bad time complexity

## What is it?

There are 2 parameters, [value, pointer]
First Node: head
Last Node: tail
They a null terminated, but an node pointing to null

## Why Linked Lists?

Loose structure and can easily add / delete nodes to between other nodes
Searching for an element is harder O(n)
Traverse because we dont know when the linked list ends 💀
The nodes are scared all over throughout the memory

prepend - O(1)
append - O(1)
lookup - O(n)
insert - O(n)
delete - O(n)

## What is a Pointer?

A reference to another place in memory such a node, value, etc...
const obj1 = {a:true};
const obj2 = obj1 // here obj2 refers to obj1 (it ain't copying, its just referring to the same position)

# Doubly Linked Lists

It has an extra piece of data which points back
Importance: Traverse our linked list from the back
Lookup can become : O(n/2) but still O(n) but a little bit more less speed

## Singly vs Doubly Linked Lists

### Singly > Doubly

Simpler implementation
Lesser Memory
Less Operations so lil bit faster

### Doubly > Singly

Great for Search
Easy to Delete

# Review

Fast Insertion
Fash Deletion
Ordered
Flexible

More Memory
Slow Lookup
