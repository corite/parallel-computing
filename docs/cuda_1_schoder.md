# Cuda 1 Schoder

## GPUs

- historically only used for rendering graphics
- nowadays general purpose (GPGPU), turing complete
- many small cores (Scalar Processor, SP)
- grouped together to multi-processors (Vector Processor)
- no branch prediction, many registers
- big & slow global memory (400-800 clock cycles access time)
- small & fast on-chip shared memory
- small caches in comparison to CPUs
- in comparison to CPUs, GPUs have more ALUs and less control units
- GPU best suited for SIMD computations

## Cuda

- many short-lived parallel threads split into 1-3 dimensions
- Warp
  - 32 threads which perform the same instruction
- Thread block
  - up to 1024 threads
  - can be synchronized
  - shared memory for data exchange
- Grid
  - groups multiple thread blocks
  - max $2^{32}-1$ in each dimensions

```C
__host__ //CPU function
__device__ //GPU function, only callable from GPU
__global__ //GPU function, callable by CPU (the kernel entry point)
cudaMalloc() //allocate GPU shared memory
cudaMemcpy() //transfer data between GPU and system memory
cudaFree() //free GPU shared memory
<<<grid,threads_per_block,sm_size,stream>>>kernel(foo) //kernel call, only grid and threads_per_block necessary
//timing
float time;
cudaEvent_t start, end;
cudaEventCreate ( &start );
cudaEventCreate ( &end );
cudaEventRecord ( start );
kernel <<< grid, threads >>>(a_dev , b_dev , size );
cudaEventRecord ( end );
cudaEventSynchronize ( end );
cudaEventElapsedTime (&time, start, end );
```