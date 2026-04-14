#include <stdio.h>
#include "trapezio.h"
#include "utils.h"

#define INTERVAL_START 0.0f
#define INTERVAL_END (2.0f * M_PI)
#define EXPECTED_RESULT -0.12212260462f
#define NUM_SUBDIVISIONS 16777216
#define TRIES 100

int main(void) {
    float a = INTERVAL_START;
    float b = INTERVAL_END;
    int n = NUM_SUBDIVISIONS;
    float h = (b - a) / n;

    int blockSize = 32;
    // We add blocksize-1 to n so that numBlocks will be zero 
    // iff n = 0.
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    float* trap_p;
    cudaMallocManaged(&trap_p, sizeof(float));
    
    double start, end;
    float result;
    int tries = TRIES;
    float errors[tries];
    double times[tries];

    printf("Calcolo integrale f(x)=x*exp(-x)*cos(2x)"
           " in [%.1f, %.1f] con %d suddivisioni.\n", 
           a, b, n);
    printf("Risultato atteso: %.11f\n\n", EXPECTED_RESULT);

    // 1. Sequenziale
    for (int t = 0; t < tries; t++) {
        start = get_cur_time();
        result = Trap_Seq(a, b, n);
        end = get_cur_time();
        errors[t] = fabs(result - EXPECTED_RESULT);
        times[t] = end - start;
    }
    printf("1. Sequenziale (CPU):\n"
           "   Err: %e | Tempo: %f sec\n\n", 
           float_avg(errors, tries), double_avg(times, tries));

    // 2. Atomic Globale
    for (int t = 0; t < tries; t++) {
        *trap_p = 0.5f * (f(a) + f(b));
        start = get_cur_time();
        Dev_trap_atomic<<<numBlocks, blockSize>>>(a, b, h, n, trap_p);
        cudaDeviceSynchronize();
        end = get_cur_time();
        result = (*trap_p) * h;
        errors[t] = fabs(result - EXPECTED_RESULT);
        times[t] = end - start;
    }
    printf("2. GPU AtomicAdd base:\n"
           "   Err: %e | Tempo: %f sec\n\n", 
           float_avg(errors, tries), double_avg(times, tries));

    // 3. Shared Tree
    for (int t = 0; t < tries; t++) {
        *trap_p = 0.5f * (f(a) + f(b));
        start = get_cur_time();
        Dev_trap_shared_tree<<<numBlocks, blockSize, 
                               blockSize * sizeof(float)>>>(a, b, h, n, trap_p);
        cudaDeviceSynchronize();
        end = get_cur_time();
        result = (*trap_p) * h;
        errors[t] = fabs(result - EXPECTED_RESULT);
        times[t] = end - start;
    }
    printf("3. GPU Shared Memory (Tree Sum):\n"
           "   Err: %e | Tempo: %f sec\n\n", 
           float_avg(errors, tries), double_avg(times, tries));

    // 4. Dissemination
    for (int t = 0; t < tries; t++) {
        *trap_p = 0.5f * (f(a) + f(b));
        start = get_cur_time();
        Dev_trap_dissemination<<<numBlocks, blockSize>>>(a, b, h, n, trap_p);
        cudaDeviceSynchronize();
        end = get_cur_time();
        result = (*trap_p) * h;
        errors[t] = fabs(result - EXPECTED_RESULT);
        times[t] = end - start;
    }
    printf("4. GPU Shared Memory (Dissemination):\n"
           "   Err: %e | Tempo: %f sec\n\n", 
           float_avg(errors, tries), double_avg(times, tries));

    // 5. Warp Shuffle
    for (int t = 0; t < tries; t++) {
        *trap_p = 0.5f * (f(a) + f(b));
        start = get_cur_time();
        Dev_trap_warp_shuffle<<<numBlocks, blockSize>>>(a, b, h, n, trap_p);
        cudaDeviceSynchronize();
        end = get_cur_time();
        result = (*trap_p) * h;
        errors[t] = fabs(result - EXPECTED_RESULT);
        times[t] = end - start;
    }
    printf("5. GPU Warp Shuffle (Tree Sum):\n"
           "   Err: %e | Tempo: %f sec\n\n", 
           float_avg(errors, tries), double_avg(times, tries));

    // 6. Multi-Warp shared memory
    for (int i = 0; i < 7; i++) {
        blockSize = blockSize+32;
        numBlocks = (n + blockSize - 1) / blockSize;
        for (int t = 0; t < tries; t++) {
            *trap_p = 0.5f * (f(a) + f(b));
            start = get_cur_time();
            Dev_trap_multi_warp<<<numBlocks, blockSize>>>(a, b, h, n, trap_p);
            cudaDeviceSynchronize();
            end = get_cur_time();
            result = (*trap_p) * h;
            errors[t] = fabs(result - EXPECTED_RESULT);
            times[t] = end - start;
        }
        printf("6. GPU Multi-Warp (blockSize: %d):\n"
               "   Err: %e | Tempo: %f sec\n\n", 
               blockSize, float_avg(errors, tries), double_avg(times, tries));
    }

    // 7. Multi-Warp shuffle
    blockSize = 32;
    for (int i = 0; i < 7; i++) {
        blockSize = blockSize+32;
        numBlocks = (n + blockSize - 1) / blockSize;
        for (int t = 0; t < tries; t++) {
            *trap_p = 0.5f * (f(a) + f(b));
            start = get_cur_time();
            Dev_trap_multi_warp_shuffle<<<numBlocks, 
                                          blockSize>>>(a, b, h, n, trap_p);
            cudaDeviceSynchronize();
            end = get_cur_time();
            result = (*trap_p) * h;
            errors[t] = fabs(result - EXPECTED_RESULT);
            times[t] = end - start;
        }
        printf("7. GPU Multi-Warp shuffle (blockSize: %d):\n"
               "   Err: %e | Tempo: %f sec\n\n", 
               blockSize, float_avg(errors, tries), double_avg(times, tries));
    }

    cudaFree(trap_p);
    return 0;
}
