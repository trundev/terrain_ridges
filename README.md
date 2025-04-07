# Geographic terrain ridge/valley generator

_Proof-of-concept tool for mountain ridge / watershed generation algorithm_
[![CI action](https://github.com/trundev/terrain_ridges/actions/workflows/main.yml/badge.svg)](https://github.com/trundev/terrain_ridges/actions/workflows/main.yml)


## Intro

The algorithm used by this tool is focused on identification of mountain ridges.
It also detects some kind of hierarchy in the ridge structure.
The same technique can be used to identify the terrain valleys, but the results are still not as useful as with the ridges.

The ridge between any two points is considered to be the line between these points, where the minimal elevation is as high as possible.
In other words, when choosing among separate lines, between the same points, the one with a single lower point will be dropped, even if its overall elevation is higher.

## Algorithm

The tool uses the following major stages:

1. Trace ridges

    Covert the source [DEM](https://en.wikipedia.org/wiki/Digital_elevation_model) data to a
    [tree-graph](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)#Tree)
    - Graph is represented by a node-grid, containing coordinates of corresponding neighbor-node
    - The neighbor node selection is similar to [Dijkstra shortest path](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
      algorithm, but the "next-hop" is selected solely by the height of "unvisited" neighbor nodes (ignore path distance)

2. Identify the "trunk" branch

    Build a branch hierarchy inside the tree
    - Assign "weight" to each node, currently coverage area
    - Group nodes into branches by accumulating these "weights" starting from the "leaf" nodes toward the "root"
    - The branch-nodes (where multiple nodes point) are attached to the "heaviest" branch,
      other branches are considered its sub-branches
    - The branch, starting at the "root" node is considered the "trunk"

3. Rearrange tree, identify all the branches

    3.1. Create a new tree-graph by flipping the "heaviest" branch:

    - The new tree has a "root" at the "leaf" node of the original tree
    - *Conjecture*: This new tree will include the heaviest possible branch, compared to other "root" node selections

    3.2. Repeat the stage 2, but keep the whole hierarchy, not just the "trunk" branch

4. Generate final geometry

    Create a vector ([GDAL OGR](https://gdal.org/en/stable/drivers/vector)) file, where:
    - Lines are along the "stem" of each branch
    - Grouped in folders (if format allows), based on corresponding branch "weight"
      compared to the [zoom level](https://wiki.openstreetmap.org/wiki/Zoom_levels) tile coverage
