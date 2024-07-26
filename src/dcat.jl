"""
    dcat(matrices)
Initialises and concatenates matrices.

Usage:
```julia-repl
dcat(matrices)
```
Arguments:
- `matrices` : matrices to concatenate.
"""
function dcat(matrices)

    # Initialize the final matrix with zeros
    n_rows = sum(size.(matrices, 1))
    n_cols = sum(size.(matrices, 2))
    final_matrix = zeros(n_rows, n_cols)

    # Row start index for each matrix
    row_start = 1
    col_start = 1

    for (i, mat) in enumerate(matrices)
        # Get the current matrix dimensions
        rows, cols = size(mat)

        # Calculate the column end index for the current block
        col_end = col_start + cols - 1

        # Place the matrix in the corresponding block
        final_matrix[row_start:(row_start + rows - 1), col_start:col_end] = mat

        # Update the row and column start indices for the next matrix
        row_start += rows
        col_start = col_end + 1
    end

    return final_matrix
end