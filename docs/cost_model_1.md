# Cost Model 1

## Metrics

- execution time
- overhead time

### Derived Metrics

- FLOPS
  - \#FLOP / execution time
- communication to computation ratio 
  - comm time / exec time
- memory access to computation ratio
  - io time / exec time

## Speed-Up

$S(p) = \frac{T(1)}{T(p)}$ where

- $p =$ number of processors
- $S(p) =$ speed-up using p processors
- $T(p) =$ execution time with p processors

## Efficiency

$E(p)=\frac{S(p)}{p}=\frac{T(1)}{p\cdot T(p)}$

## Goal

$S(p)\approx p, E(p) \approx 1$

## Ahmdals Law

$S(v) = \frac{1}{h+\frac{f}{v}}$

where

- $h=1-f=$ serial part of code
- $f =$ parallel part of code
- $v =$ speedup for parallel part

Problems with this definition:

- implicitly assumes that, when one is interested in speedup, the problem size does not change
  - this is almost never the case. If an algorithm gets faster, it is usually applied to bigger problem size
  - it is more realistic to assume that the overall computation time stays the same
- in reality, the serial parts of the code are usually, startup, IO, ... tasks
  - do not scale with problem size
- in summary, Ahmdals Law is too pessimistic $\rightarrow$ Gustafson's Law

## Gustafson's Law

$S_g(p)=\frac{h'+f'\cdot p}{h'+f'}$

$S_g(p)=h'+f'\cdot p$

$S_g(p)=h'+ (1-h')\cdot p$

$S_g(p)=p+(1-p)\cdot h'$

## Iso Efficiency

Iso-efficiency function is a model that relates the size of the problem being
solved to the number of processors required to maintain the efficiency at a
fixed value - “Iso” from Ancient Greek ἴσος (ísos, “equal”).

- $W(n)$ Problem size – function on input size n describing the total number of basic operations to solve a problem
- $T(1,W(n))$ Run-time of problem for 1 processor depending on problem size
- $T_O(p,W(n))= p\cdot T(p,W(n))-T(1,W(n))$ Total Overhead function – function depending on the number of processing units and the problem size
- $T(p,W(n)) = \frac{T(1,W(n))+T_O(p,W(n))}{p}$ (Parallel) run-time of problem – depending on the number of processing units and the problem size
- $S(p,W(n))=\frac{T(1,W(n))}{T(p,W(n))}$ the resulting speedup
- $E(p,W(n))=\frac{S(p,W(n))}{p}$ the efficiency
- $E^*$ the desired efficiency
- $\kappa = \frac{E^*}{1-E^*}$ then $W(n)=\kappa*T_O(p,W(n))$

## Roofline Model

Simple model to determine if an application is bound by peak band-width or peak performance.
The model is based on arithmetical/operational intensity measuring the number of floating-point operations/operations per byte.

- peak performance $\pi$: FLOPS
- operational intensity $I$: $\frac{\# FLOP}{\# byte}$
- memory bandwith $\beta$: $\frac{\# byte}{sec}$
- $P=\min(\frac{\pi}{\beta\cdot I})$ the actual achievable performance of this code

## Matrix Vector

- Stripe based: each processor has some rows in the matrix and the corresponding vector entries assigned
- checkerboard:
