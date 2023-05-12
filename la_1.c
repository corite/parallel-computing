#include <stdio.h>
#include <malloc.h>

double **alloc_mat(int col, int row) {
    double *actual_mat = malloc(col * row * sizeof(double));
    double **mat = malloc(row * sizeof(double *));
    for (int i = 0; i < row; i++)
        mat[i] = actual_mat + (i * col * sizeof(double));
    return mat;
}

void free_mat(double **A) {
    free(*A);
    free(A);
}

void alloc_spmat(int nnz, int **col, int **row, double **A) {
    *col = malloc(nnz * sizeof(int));
    *row = malloc(nnz * sizeof(int));
    *A = malloc(nnz * sizeof(double));
}

void outer_product(int n, int m, double *u, double *v, double **A) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j)
            A[i][j] = u[i] * v[j];
    }
}

void sparsemv(int n1, int n2, int nnz, int *r, int *c, double *a, double *x, double *b) {
    //i assume b is initialised with zeros
    for (int i = 0; i < nnz; ++i) {
        b[r[i]] += a[i] * x[c[i]];
    }
}

void matrix_vector(int n1, int n2, double **A, double *x, double *b) {
    for (int i = 0; i < n1; ++i) {
        b[i] = 0;
        for (int j = 0; j < n2; ++j)
            b[i] += A[i][j] * x[j];
    }
}

void test_sparsemv() {
    double *A;
    int *col;
    int *row;
    alloc_spmat(6, &col, &row, &A);
    A[0] = 3;
    col[0] = 1;
    row[0] = 0;
    A[1] = 3;
    col[1] = 0;
    row[1] = 1;
    A[2] = 3;
    col[2] = 2;
    row[2] = 1;
    A[3] = 3;
    col[3] = 0;
    row[3] = 2;
    A[4] = 3;
    col[4] = 1;
    row[4] = 2;
    A[5] = 3;
    col[5] = 2;
    row[5] = 2;
    double *x = malloc(3 * sizeof(double));
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;
    double *b = calloc(3 , sizeof(double));
    sparsemv(0, 0, 6, row, col, A, x, b);
    for (int i = 0; i < 3; ++i) {
        printf("%f\n", b[i]);
    }
}

int main() {
    test_sparsemv();
    return 0;
}

void test_matrix_vector() {

}


