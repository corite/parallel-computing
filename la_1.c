#include <stdio.h>
#include <malloc.h>

double **alloc_mat(int col, int row) {
    double *actual_mat = malloc(col * row * sizeof(double));
    double **mat = malloc(row * sizeof(double*));
    for (int i = 0; i < row; i++)
        mat[i] = actual_mat+ (i*col* sizeof(double));
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

int main() {
    int nnz = 2;
    int *col;
    int *row;
    double *val;
    alloc_spmat(nnz, &col, &row, &val);
    col[0] = 0;
    row[0] = 0;
    col[1] = 1;
    row[1] = 1;
    val[0] = 1;
    val[1] = 2;

    double vec[] = {2.0, 3.0};
    double res[] = {0.0,0.0};
    sparsemv(2,2,2,row,col,val,vec,res);
    printf("%f,%f", res[0], res[1]);

    return 0;
}

void test_matrix_vector() {

}


