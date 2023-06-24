#include <stdio.h>
#include <omp.h>

#define MAX_THREADS 6
static const long num_steps = 100000000;
double step = 1.0 / (double) num_steps;

int main() {
    double pi, full_sum;
    double start_time, run_time;
    for (int num_threads = 1; num_threads <= MAX_THREADS; num_threads++) {
        omp_set_num_threads(num_threads);
        full_sum = 0.0;
        start_time = omp_get_wtime();
        printf("num_threads = %d\n", num_threads);
#pragma omp parallel
        {
            int id = omp_get_thread_num();
            double x;
            double partial_sum = 0;
            for (int i = id; i < num_steps; i += num_threads) {
                x = (i + 0.5) * step;
                partial_sum += +4.0 / (1.0 + x * x);
            }
#pragma omp atomic update
            full_sum += partial_sum;
        }
        pi = step * full_sum;
        run_time = omp_get_wtime() - start_time;
        printf("pi is %f in %f seconds %d threads\n",
               pi, run_time, num_threads);
    }
    return 0;

}