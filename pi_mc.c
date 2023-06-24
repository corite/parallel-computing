#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_THREADS 16

static inline double my_rand(int *seeds) {
    return (double) rand_r(seeds + omp_get_thread_num()) / RAND_MAX;
}

void mc(int N) {
    int I = 0;
    double start = omp_get_wtime();
    int *seeds = malloc(NUM_THREADS * sizeof(int));
    for (int i = 0; i < NUM_THREADS; ++i) {
        seeds[i] = i;
    }


#pragma omp parallel for reduction(+:I) num_threads(NUM_THREADS)
    for (int i = 0; i < N; i++) {
        double x = my_rand(seeds);
        double y = my_rand(seeds);
        if (sqrt(x * x + y * y) <= 1) I++;
    }
    double pi = (double) I / N * 4;
    double end = omp_get_wtime();
    printf("%f,%f\n", pi, end - start);
}


int main() {
    for (int i = 1; i < 10000000; i *= 10) {
        mc(i);
    }
    return 0;
}

