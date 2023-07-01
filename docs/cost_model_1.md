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

- p = number of processors
- S(p) = speed-up using p processors
- T(p) = execution time with p processors

## Efficiency

$E(p)=\frac{S(p)}{p}=\frac{T(1)}{p\cdot T(p)}$

## Goal

$S(p)\equiv p, E(p) \equiv 1$

## Ahmdals Law

$S(v) = \frac{1}{h+\frac{f}{v}}$

where

- h=1-f= serial part of code
- f = parallel part of code
- v = speedup for parallel part (f)

## Gustafson's Law

(Ahmdal is to pessimistic)

$S_g(p)=p+(1-p)\cdot h$

## Iso Eficciency

Iso-efficiency function is a model that relates the size of the problem being
solved to the number of processors required to maintain the efficiency at a
fixed value - “Iso” from Ancient Greek ἴσος (ísos, “equal”).

- $W(n)$ Problem size – function on input size n describing the total number of basic operations to solve a problem
- $T(1,W(n))$ Run-time of problem for 1 processor depending on problem size
- $T(p,W(n))$ (Parallel) run-time of problem – depending on the number of processing units and the problem size
- $T_O(p,W(n))$ Total Overhead function – function depending on the number of processing units and the problem size

//todo what else?

## Roofline Model

Simple model to determine if an application is bound by peak band-width or peak performance.
The model is based on arithmetical/operational intensity measuring the number of floating-point operations/operations per byte.

//todo what else?
