# Cuda 1

```C
int tid = threadIdx.x;
int col_offset = blockDim.x * blockDim.y * blockIdx.x;
int row_offset = gridDim.x * blockIdx.y * blockDim.x * blockDim.y + blockDim.x * threadIdx.y;
int gridid = tid + col_offset + row_offset;
```

see [cuda schoder](cuda_schoder.md), since it has pretty much the same content.