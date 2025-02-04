## Introduction

Nodes are either red or black
Leaves are black
If node a is red, the childeren are black
All paths from a node to its NIL ()
Nodes require one storage bit to keep track of color

## Rotations

Caused due to Insert and Remove operations

1. aletrs the structure of a tree by rearranging subtrees
2. goal is to decrease the height of the tree
   - dont effect the order of the elements

## Insertions

### Strategy

Insert Z and color it red
Recolor and rotate nodes to fix violation

1. Z = root -> color black
2. z.uncle = red -> recolor
3. Z.uncle = black(traingle) -> rotate.z.parent
4. Z.uncle = black(link) -> rotate.z.grandparent and also recolor

### Examples
