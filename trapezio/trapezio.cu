#include "trapezio.h"
#include <math.h>

// --------------------------------------------------------
// FUNZIONE MATEMATICA f(x)
// --------------------------------------------------------
__host__ __device__ float f(float x) {
    return x * expf(-x) * cosf(2.0f * x);
}

// --------------------------------------------------------
// 1. ALGORITMO SEQUENZIALE (HOST)
// --------------------------------------------------------
float Trap_Seq(float a, float b, int n) {
    float h = (b - a) / n;
    float somma = f(a) + f(b);
    for (int i = 1; i < n; i++) {
        float x_i = a + i * h;
        somma += 2.0f * f(x_i);
    }
    return (h / 2.0f) * somma;
}

// --------------------------------------------------------
// 2. KERNEL BASE CON ATOMICADD
// --------------------------------------------------------
__global__ void Dev_trap_atomic(const float a, const float b, const float h, const int n, float* trap_p) {
    int my_i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (my_i > 0 && my_i < n) {
        float my_x = a + my_i * h;
        float my_trap = f(my_x);
        atomicAdd(trap_p, my_trap); 
    }
}

// --------------------------------------------------------
// 3. TREE-STRUCTURED SUM
// --------------------------------------------------------
__device__ float Shared_mem_tree_sum(float* sdata) {
    int tid = threadIdx.x;
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
    }
    return sdata[0];
}

__global__ void Dev_trap_shared_tree(const float a, const float b, const float h, const int n, float* trap_p) {
    extern __shared__ float sdata[];
    
    int my_i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    
    sdata[tid] = (my_i > 0 && my_i < n) ? f(a + my_i * h) : 0.0f;
    
    float block_sum = Shared_mem_tree_sum(sdata);
    
    if (tid == 0) {
        atomicAdd(trap_p, block_sum);
    }
}

// --------------------------------------------------------
// 4. DISSEMINATION SUM
// --------------------------------------------------------
__device__ float Shared_mem_dissemination_sum(float sval[]) {
    int tid = threadIdx.x;
    int mylane = tid % WARP_SIZE;
    int warp_start = tid - mylane;
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        int source = warp_start + ((mylane + offset) % WARP_SIZE);
        
        float val_to_add = sval[source];
        
        __syncwarp(); 
        
        sval[tid] += val_to_add;
        
        __syncwarp(); 
    }
    
    return sval[tid];
}

__global__ void Dev_trap_dissemination(const float a, const float b, const float h, const int n, float* trap_p) {
    __shared__ float sval[WARP_SIZE];
    
    int my_i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    
    sval[tid] = (my_i > 0 && my_i < n) ? f(a + my_i * h) : 0.0f;
    
    float warp_sum = Shared_mem_dissemination_sum(sval);
    
    if (tid % WARP_SIZE == 0) {
        atomicAdd(trap_p, warp_sum);
    }
}


// --------------------------------------------------------
// 5. WARP SHUFFLE
// --------------------------------------------------------
__device__ float Warp_Sum(float val) {
    unsigned mask = 0xFFFFFFFF;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void Dev_trap_warp_shuffle(const float a, const float b, const float h, const int n, float* trap_p) {
    int my_i = blockDim.x * blockIdx.x + threadIdx.x;
    int mylane = threadIdx.x % WARP_SIZE;
    
    float my_val = (my_i > 0 && my_i < n) ? f(a + my_i * h) : 0.0f;
    float warp_sum = Warp_Sum(my_val);
    
    if (mylane == 0) {
        atomicAdd(trap_p, warp_sum);
    }
}

// --------------------------------------------------------
// 6. MULTI-WARP OPTIMIZED
// --------------------------------------------------------
__global__ void Dev_trap_multi_warp(const float a, const float b, const float h, const int n, float* trap_p) {
    __shared__ float thread_calcs[MAX_BLKSZ]; 
    __shared__ float warp_sum_arr[WARP_SIZE];   

    int tid = threadIdx.x;
    int w = tid / WARP_SIZE;
    int my_lane = tid % WARP_SIZE;
    int my_i = blockDim.x * blockIdx.x + tid;

    thread_calcs[tid] = (my_i > 0 && my_i < n) ? f(a + my_i * h) : 0.0f;
    __syncthreads();

    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
        if (my_lane < stride) {
            thread_calcs[tid] += thread_calcs[tid + stride];
        }
        __syncwarp();
    }

    if (my_lane == 0) {
        warp_sum_arr[w] = thread_calcs[tid];
    }
    __syncthreads(); 

    if (w == 0) {
        int num_warps = blockDim.x / WARP_SIZE;
        
        if (my_lane >= num_warps) {
            warp_sum_arr[my_lane] = 0.0f;
        }
        __syncwarp();

        for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
            if (my_lane < stride) {
                warp_sum_arr[my_lane] += warp_sum_arr[my_lane + stride];
            }
            __syncwarp();
        }

        if (my_lane == 0) {
            atomicAdd(trap_p, warp_sum_arr[0]);
        }
    }
}