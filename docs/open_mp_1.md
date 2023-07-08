# OpenMP 1

## Parallelization Types

### Distributed System

- processors have local memory, communication via network
- data movement expensive
- easily scalable

### Shared Memory System

- communication via memory
- data movement "cheap" (not really though)
- not very scalable

## OpenMP Syntax

```C
#pragma omp {parallel|single|master}
#pragma omp parallel [num_threads(i)] [private(vars)] [shared(vars)] [default(none|shared|private)] [if(expr)] //default none means that all vars needed to have an explicit scope, nothing is assumed
#pragma omp parallel for {private|shared|default}(vars) reduction(op:var)//define scope of variables used in for loop, they need to be defined beforehand. Specify reduction p.e. (+:a)
#pragma omp parallel for {firstprivate|lastprivate}(vars) //defines how vars are initialized and which value is returned to outside of the parallel region. A variable can be both firstprivate and lastprivate

omp_set_num_threads(i);
omp_get_num_threads();
omp_get_thread_num();
omp_get_max_threads();
omp_get_wtime();
```