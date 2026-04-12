#ifndef TRAPEZIO_H
#define TRAPEZIO_H

#define WARP_SIZE 32
#define MAX_BLKSZ 1024

// Funzione matematica da integrare
__host__ __device__ float f(float x);

// Implementazione sequenziale su Host
float Trap_Seq(float a, float b, int n);

// Kernel CUDA
__global__ void Dev_trap_atomic(const float a, const float b, const float h, const int n, float* trap_p);
__global__ void Dev_trap_shared_tree(const float a, const float b, const float h, const int n, float* trap_p);
__global__ void Dev_trap_dissemination(const float a, const float b, const float h, const int n, float* trap_p);
__global__ void Dev_trap_warp_shuffle(const float a, const float b, const float h, const int n, float* trap_p);
__global__ void Dev_trap_multi_warp(const float a, const float b, const float h, const int n, float* trap_p);

#endif // TRAPEZIO_H