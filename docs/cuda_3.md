# Cuda 3

## NVPROF

- profiling tool for Nvidia GPUs
- alternative: Nvidia NSight

## Host Memory: Paged vs Pinned

- in modern OSs, memory is organized in same-sized pages
- when the OS needs more memory than is physically available, it can write less frequently used pages to persistent storage, thereby effectively freeing memory
- problem: access to these swapped-out pages is now very slow
- to prevent the os to swap out a certain page, it can be pinned (page-locked)
- pinned memory can, in contrast to to pageable memory, be accessed directly by the GPU (Question: does this circumvent the whole memory virtualization process, i.e. can the OS move pinned pages for p.e. defragmentation)
- accessing pinned memory is generally faster, but allocating too much of it can slow down the whole system since it has less overall memory available

```C
cudaHostAlloc(ptr,sze,flg); //allocate pinned memory 
cudaFreeHost(ptr); //free pinned memory
```

## Zero-Copy Memory

- enables the GPU to directly access host memory
- mostly beneficially for integrated GPUs, since there CPU/GPU memory is physically shared
- with discrete GPUs only beneficial in some cases (mostly write/read once), since the feature disables GPU caching

## Unified Memory (Unified Virtual Addressing)

- host and devices share a single virtual address space

## Streams

A stream is a sequence of commands (possibly issued by different host threads) that execute in order. Different streams, on the other hand, may execute their commands out of order with respect to one another or concurrently.

```C
cudaStreamCreate(id); // create a new stream with a specified id
kernel<<<blocks,threads,0,id>>>(); // execute a kernel on a given stream
cudaStreamSynchronize(id|void);// blocks host until all computations on the specified stream have completed. If no stream is specified, the host waits on all streams
cudaStreamDestroy(id); //delete stream
cudaStreamWaitEvent(id,event,flags); // synchronize stream with a given event
```

## Asynchronous Copy

```C
cudaMemcpyAsync(dst,src,sze,mod); // async copy memory
```