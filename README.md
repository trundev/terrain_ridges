# Terrain ridge/valey generator
_Proof-of-concept tool for mountain ridge / watershed generation algorithm_

## Into
The algorithm used by this tool is focused on identification of mountain ridges. It also detects some kind of hierarchy in the ridge structure.
The same technique can be used to identify the terrain valleys, but the results are still not as useful as with the ridges.

The ridge between any two points is considered to be the line between these points, where the minimal elevation is as higher as possible.
In other words, when choosing among separate lines, between the same points, the one with a single low point will be dropped, even if its overall elevation is higher.
