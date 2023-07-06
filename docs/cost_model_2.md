# Cost Model 2

## Pipelining

I think this is just a theoretical example, the point of which I fail to grasp.
    
## HyperThreading

- problem: 
  - threads tend to do a lot of IO, during which the processor can't do anything except wait
  - doing a full context switch (to a different to a different thread) every time is pretty expensive
- solution:
  - introduce virtual processors (usually 2x physical processors)
  - OS distributes work to the virtual processors, as if they were real
  - the physical processor thereby always has two threads assigned to it
  - the hardware directly supports fast context switches between these two threads

## Race Conditions

- Shit happens. Deal with it.

## Bernstein Condition

- formalization of when tasks are independent of each other
- helps avoid race conditions and find parallel structures
- $I_1\cap O_2 = \emptyset,I_2\cap O_1 = \emptyset, O_1\cap O_2 = \emptyset$

## Task Graph

- directed graph
- vertices are tasks
- edges are data flows
- critical path: all paths with maximum length inside the graph
