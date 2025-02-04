# Breadth First Search (BFS)

An algorithm for searching a graph
Breadth means broad / wide
We use a queue to track verttices (the ones to be visited)
Time Complexity: O(|V| + |E|) -> (Vertices + Edges)

# Depth First Search (DFS)

Follows one branch of a tree until the target node or the end is reached then it goes back and goes to the next branch.
Lower Memory requirements, because we dont need to keep child pointers
We need to go deep as possible and then start going to the right until the traversal of the tree is completed

## Implementation methods

// 9
// 4 20
// 1 6 15 170
PreOrder - [9,4,1,6,20,15,170] // Used to recreate trees
InOrder - [1,4,6,9,15,20,170]
PostOrder - [1,6,4,15,170,20,9]

# BFS vs DFS

Breadth first search is like water flooding from the top, breadth first search is like water flooding in streams from the top and going to the right.
They both do the same thing but they are used for different reasons.

## BFS

Shortest Path, Closer Node: Pros
More Memory: Cons

If an value is in the top of the tree then this is useful because this gets to each level first

## DFS

Less Memory, Does Path Exist?: Pros
Can Get Slow: Cons

# Graph Traversals

[Visual Graph Traversal](https://visualgo.net/en/dfsbfs)
BFS - Shortest Path
DFS - Check to see if something exists

## BFS

The unique thing is that it lands it self to be able to find the shortest path between certain nodes. Used in peer to peer networks, google maps.

## DFS

This is the same concept of solving a maze, it is exactly the same. That is why we use recursion, the concept of backtracking is harder to do with iterations, and it is build in the fundamentals of recursion

# Dijkstra + Bellman Ford Algorithms

Allow us to find the shortest path between 2 nodes of a weighted graph.

## Dijkstra

Take less time than bellman ford
Worst Case: O(e log v)

[Dijkstra's algorithm in 3 minutes](https://www.youtube.com/watch?v=_lHSawdgXpI&ab_channel=MichaelSambol)
[Graph Data Structure 4. Dijkstra’s Shortest Path Algorithm](https://www.youtube.com/watch?v=pVfj6mxhdMw&ab_channel=ComputerScience)
[Dijkstra's Shortest Path Algorithm | Graph Theory](https://www.youtube.com/watch?v=pSqmAO-m7Lk&ab_channel=WilliamFiset)
[How Dijkstra's Algorithm Works](https://www.youtube.com/watch?v=EFg3u_E6eHU&ab_channel=SpanningTree)
[Implement Dijkstra's Algorithm](https://www.youtube.com/watch?v=XEb7_z5dG3c&ab_channel=NeetCodeIO)

## Bellman Ford

Can accommodate negative weights
Worst Case: O(n^2)

[Bellman-Ford in 4 minutes — Theory](https://www.youtube.com/watch?v=9PHkk0UavIM&ab_channel=MichaelSambol)
[Bellman-Ford in 5 minutes — Step by step example](https://www.youtube.com/watch?v=obWXjtg0L64&ab_channel=MichaelSambol)
[Bellman Ford Algorithm | Shortest path & Negative cycles | Graph Theory](https://www.youtube.com/watch?v=lyw4FaxrwHg&ab_channel=WilliamFiset)
