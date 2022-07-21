# Graph Algorithms
C++ implementation of useful graph primitives. This program implements a sparse graph class, with the most well known algorithmic primitives useful for graph problems. 
The program supports the basic algorithms (DFS, BFS), shortest paths finders (single-source and all-pairs), Kosaraju's algorithm for strongly connected components and other 
graph utilities (reverse a graph, construct a random graph etc.)

# Example use
Suppose we need a function to topologically sort a graph. When dfs traversing the graph vertex will be finished in a reverse topological ordering, so we only need to pass a callback that stores the vertices in that order:

    stack<int> sorted;
    our_graph.dfs([](int v){}, [&](int v){sorted.push(v);});

# Features
* Our DFS accepts functional values for routines to be executed at vertex discovery and after vertex exploration. This allows us to afterwards reuse the code much more easily (e.g. for strongly connected components)
* The random_graph function can create diverse examples of graphs which we can be very useful for debugging graph algorithms.
* Style note: We use mostly recursive functions for clarity, and pass state vectors (such as visits vectors) by reference, making reasonably efficient memory use.
