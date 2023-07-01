# Linear Algebra 2

## Matrix-Matrix Multiplication

$C=A\times B$

### Naive

- compute every entry of $C$ one by one (can be done in parallel)
- $O(n^3)$

## Block

- same as naive, but combines multiple entries to one block
- still $O(n^3)$, but better data locality

## Strassen

- partition matrices in 4 blocks
- compute some intermediate matrices $M_1-M_7$
- less multiplications needed $O(n^{\log_27})\approx O(n^{2.807})$

## Parallel Block Matrix Product

- Block, but in parallel...

## Cannon's Matrix Product

- Problem with former: to compute one block of $C$, every parallel units needs 3 blocks of both $A$ and $B$ $\rightarrow$ memory inefficient
- solution: at a time, one parallel unit only has one block of $A$ and $B$
- $C$ is computed incrementally, by passing the blocks of $A$ and $B$ around

## DNS Algorithm

- given $n^3$ processors, this approach can achieve $O(\log n)$ parallel runtime
- not performant in reality because of slow interprocessor communication