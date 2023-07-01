# Linear Algebra 1

- Scalar, Vector, Tensor...
- Float, Char, Double...

## Dense Matrices

- every entry has a value in memory (even 0)
- can be written consecutively
- column major vs. row major

## Sparse Matrices

- most entries are 0 and should not take up space in memory

### Coordinate-List-Representation

- for each non-zero entry, save row, column and value
- all other entries are implicitly zero

### Compressed-Row-Format-Representation

- like previous, but entries sorted by row, then column
- therefore each row only has to be saved once (and not for every entry in that row)
- when going through the list of column-value entries, the next corresponding row entry is chosen when the column decreases, otherwise it stays the same

## Applications

- graph algorithms (navigation, search)
- ML
