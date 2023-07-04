# OpenMP 2

## False Sharing

- different threads use elements of the same cacheline
- this prevents the line to move into a faster cache, harming performance

## Synchronization

```C
#pragma omp critical //block executed by one thread at a time
#pragma omp atomic [update|read|write|capture] //make next memory update atomic, might be more performant than 'critical'
#pragma omp barrier //wait until all threads have arrived
#pragma omp [for] nowait //remove implicit barrier at the end of for
#pragma omp [for] ordered //preserve iteration order
```
## SMP / NUMA

### SMP (Symmetric Multiprocessor)

- 1 shared memory
- 1 or more equivalent processors
- depending on who you ask, caches are either allowed in SMP or they are already a NUMA concept 

### NUMA (Non-uniform memory access)

- every processor gets a certain local region of memory assigned (additionally there might also be a shared memory)
- this region should be close to the processor
- local memory is faster than global
- downside: moving memory between processes is costly

## Tasks

//todo
