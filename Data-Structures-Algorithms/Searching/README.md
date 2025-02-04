# Introduction

Searching is something we do a lot in our computers, searching we use everyday such as finding files in our computer or finding using Ctrl + F. Searching is a big part of our life's but how are the computer we use able to search in a such a small amount of time.
Covering:

- Linear Search
- Binary Search
- Depth First Search
- Breadth First Search

# Linear Search

Method of finding an element in a list, it sequentially check for every element in the list until the value is found.
It is the type of search that is usually build into programming languages

Best Case: O(1)
Worst Case: O(n)

# Binary Search

The list must be sorted, and what we do is check the middle element and then see if the middle element is higher or lower than the value we are looking for and then we divide the list and search in the specific part we need to until we find the element we want to. This is kind of the same implementation as the divide and concur way of quick and merge sort
This is the same way we implement binary search in a binary tree.

Time Complexity: O(log n)

# Traversals (Graph + Tree)

When we want to do something to each node in a tree or graph, or we dont have a sorted tree, so we will make it sorted.
We can think as if we are visting every node so its O(n). We can use loops for this cause aswell but we are going to do it in a different manner when it comes to graphs and trees. Options:

- Breadth First Search (BFS)
- Depth Fist Search (DFS)
