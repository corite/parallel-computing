# Linear Algebra 1

- Scalar, Vector, Tensor...
- Float, Char, Double...

## Dense Matrices

- every entry has a value in memory (even 0)
- can be written consecutively
- column major vs. row major

## Sparse Matrices

- most entries are 0 and should not take up space in memory
- $\rightarrow$ only non-zero entries are saved

### Coordinate-List-Representation

- for each non-zero entry, save row, column and value
- all other entries are implicitly zero

### Compressed-Row-Format-Representation

- like previous, but entries sorted by row, then column
- the row array now always points to the beginning of the row in the value array
- pro: allows fast row access and saves memory

## Applications

- graph algorithms (navigation, search)
- ML
