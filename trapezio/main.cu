#include <stdio.h>
#include "trapezio.h"
#include "utils.h"

#define INTERVAL_START 0.0f
#define INTERVAL_END (2.0f * M_PI)
#define EXPECTED_RESULT -0.12212260462f
#define NUM_SUBDIVISIONS 10000000

int main(void) {
    float a = INTERVAL_START;
    float b = INTERVAL_END;
    int n = NUM_SUBDIVISIONS;
    float h = (b - a) / n;

    int blockSize = 32;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    float* trap_p;
    cudaMallocManaged(&trap_p, sizeof(float));
    
    double start, end;
    float result;

    printf("Calcolo integrale f(x)=x*exp(-x)*cos(2x) in [%.1f, %.1f] con %d suddivisioni.\n", a, b, n);
    printf("Valore atteso: %.11f\n\n", EXPECTED_RESULT);

    // 1. Sequenziale
    start = get_cur_time();
    result = Trap_Seq(a, b, n);
    end = get_cur_time();
    printf("1. Sequenziale (CPU):\n   Risultato: %f (Err: %e) | Tempo: %f sec\n\n", result, fabs(result - EXPECTED_RESULT), end - start);

    // 2. Atomic Globale
    *trap_p = 0.5f * (f(a) + f(b));
    start = get_cur_time();
    Dev_trap_atomic<<<numBlocks, blockSize>>>(a, b, h, n, trap_p);
    cudaDeviceSynchronize();
    end = get_cur_time();
    result = (*trap_p) * h;
    printf("2. GPU AtomicAdd base:\n   Risultato: %f (Err: %e) | Tempo: %f sec\n\n", result, fabs(result - EXPECTED_RESULT), end - start);

    // 3. Shared Tree
    *trap_p = 0.5f * (f(a) + f(b));
    start = get_cur_time();
    Dev_trap_shared_tree<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(a, b, h, n, trap_p);
    cudaDeviceSynchronize();
    end = get_cur_time();
    result = (*trap_p) * h;
    printf("3. GPU Shared Memory (Tree Sum):\n   Risultato: %f (Err: %e) | Tempo: %f sec\n\n", result, fabs(result - EXPECTED_RESULT), end - start);

    // 4. Dissemination
    *trap_p = 0.5f * (f(a) + f(b));
    start = get_cur_time();
    Dev_trap_dissemination<<<numBlocks, blockSize>>>(a, b, h, n, trap_p);
    cudaDeviceSynchronize();
    end = get_cur_time();
    result = (*trap_p) * h;
    printf("4. GPU Shared Memory (Dissemination):\n   Risultato: %f (Err: %e) | Tempo: %f sec\n\n", result, fabs(result - EXPECTED_RESULT), end - start);

    // 5. Warp Shuffle
    *trap_p = 0.5f * (f(a) + f(b));
    start = get_cur_time();
    Dev_trap_warp_shuffle<<<numBlocks, blockSize>>>(a, b, h, n, trap_p);
    cudaDeviceSynchronize();
    end = get_cur_time();
    result = (*trap_p) * h;
    printf("5. GPU Warp Shuffle (Tree Sum):\n   Risultato: %f (Err: %e) | Tempo: %f sec\n\n", result, fabs(result - EXPECTED_RESULT), end - start);

    // 6. Multi-Warp Optimized
    for(int i=0;i<7;i++) {
        *trap_p = 0.5f * (f(a) + f(b));

        blockSize = blockSize+32;
        numBlocks = (n + blockSize - 1) / blockSize;

        start = get_cur_time();

        Dev_trap_multi_warp<<<numBlocks, blockSize>>>(a, b, h, n, trap_p);

        cudaDeviceSynchronize();
        end = get_cur_time();
        result = (*trap_p) * h;

        printf("6. GPU Multi-Warp (blockSize: %d):\n   Risultato: %f (Err: %e) | Tempo: %f sec\n\n", blockSize, result, fabs(result - EXPECTED_RESULT), end - start);
    }

    cudaFree(trap_p);
    return 0;
}