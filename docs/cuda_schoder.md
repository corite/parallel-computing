# Cuda Schoder

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
cudaMalloc(ptr,sze); //allocate GPU shared memory
cudaMemcpy(dst,src,sze,mod); //transfer data between GPU and system memory, mod \in cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault (inferred)
cudaFree(ptr); //free GPU shared memory
<<<dim3_grid,dim3_block,int_dyn_mem_p_block,stream>>>kernel(foo) //kernel call, only dim3_grid and dim3_block necessary
<<<int_blocks,int_threads_per_block>>>kernel(foo) //kernel call, like previous but squashes everything into 1 dimension (I think)
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
[kernel parameters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)

## Execution

```bash
nvcc -o beispiel beispiel.cu -arch=sm_35

```
