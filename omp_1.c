#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//////////////////////// Helper Functions

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

//////////////////////// Actual code

int my_inner_par(int n, int *u, int *v) {
    int b = 0;
#pragma omp parallel for reduction(+:b)
    for (int j = 0; j < n; ++j) {
        b += u[j] * v[j];
    }
    return b;
}

void matrix_vector_par(int n1, int n2, double **A, double *x, double *b) {
    for (int i = 0; i < n1; i++) {
#pragma omp parallel for reduction(+:b[i])
        for (int j = 0; j < n2; j++) {
            b[i] += A[i][j] * x[j];
        }
    }
}

void outer_product_par(int n, int m, double *u, double *v, double **A) {
    for (int i = 0; i < n; ++i) {
#pragma omp parallel for firstprivate(i)
        for (int j = 0; j < m; ++j)
            A[i][j] = u[i] * v[j];
    }
}

void sparsemv_par(int n1, int n2, int nnz, int *r, int *c, double *a, double *x, double *b) {
    int i;
#pragma omp parallel for reduction(+:b[r[i]])
    for (i = 0; i < nnz; i++) {
        b[r[i]] += a[i] * x[c[i]];
    }
}

//////////////////////// Tests

int my_inner_par_test() {
    printf("my_inner_par_test\n");
    int n = 1000;
    double *time = malloc(n * sizeof(double));
    FILE *fp;
    fp = fopen("my_inner_par_test.txt", "w+");

    for (int i = 0; i < n; ++i) {
        int *u = malloc(i * sizeof(int));
        int *v = malloc(i * sizeof(int));
        for (size_t j = 0; j < i; ++j) {
            u[j] = 1;
            v[j] = j;
        }

        double start = omp_get_wtime();
        int b = my_inner_par(i, u, v);
        double end = omp_get_wtime();
        time[i] = end - start;
        fprintf(fp, "%d,%f\n", i, time[i]);
    }
    return 0;
}

int matrix_vector_par_test() {
    printf("matrix_vector_par_test\n");
    int n = 8;
    FILE *fp;
    fp = fopen("matrix_vector_par_test.txt", "w+");

    for (int i = 1; i < n; ++i) {
        double *b = malloc(i * sizeof(double));
        double *x = malloc(i * sizeof(double));
        for (int j = 0; j < i; ++j) {
            x[j] = 1.0;
        }
        double **A = alloc_mat(i, i);
        for (int s = 0; s < i; s++) {
            for (int t = 0; t < i; t++) {
                A[s][t] = 1.0;
            }
        }

        double start = omp_get_wtime();
        matrix_vector_par(i, i, A, x, b);
        double end = omp_get_wtime();
        double time = end - start;
        fprintf(fp, "%d,%f\n", i, time);
        free_mat(A);
        free(x);
        free(b);

    }
    return 0;
}

int outer_product_par_test() {
    printf("outer_product_par_test\n");
    int n = 10;
    FILE *fp;
    fp = fopen("outer_product_par_test.txt", "w+");

    for (int i = 1; i < n; ++i) {
        double *u = malloc(i * sizeof(double));
        double *v = malloc(i * sizeof(double));
        for (int j = 0; j < i; ++j) {
            u[j] = 1.0;
            v[j] = (double) j;
        }
        double **A = alloc_mat(i, i);

        double start = omp_get_wtime();
        outer_product_par(i, i, u, v, A);
        double end = omp_get_wtime();
        double time = end - start;
        fprintf(fp, "%d,%f\n", i, time);
        free_mat(A);
        free(u);
        free(v);

    }
    return 0;
}

int sparsemv_par_test() {
    printf("sparsemv_par_test\n");
    //TODO
    return 0;
}

int main() {
    my_inner_par_test();
    matrix_vector_par_test();
    outer_product_par_test();
    sparsemv_par_test();
    return 0;
}

