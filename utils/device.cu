#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("Number of CUDA devices: %d\n", devCount);

    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);

        printf("\nDevice %d: %s\n", i, props.name);
        printf("  Compute Capability: %d.%d\n", props.major, props.minor);
        printf("  Number of SMs: %d\n", props.multiProcessorCount);
        printf("  Total Global Memory: %zu MB\n", props.totalGlobalMem / (1024 * 1024));
        printf("  Memory Bus Width: %d bits\n", props.memoryBusWidth);
        printf("  Memory Type: %s\n", props.memoryClockRate > 0 ? "GDDR" : "Unknown");
        printf("  L2 Cache Size: %d KB\n", props.l2CacheSize / 1024);
        printf("  Texture Alignment: %zu bytes\n", props.textureAlignment);

        CUresult res = cuInit(i);
        if (res != CUDA_SUCCESS) {
            printf("CUDA Driver API initialization failed: %d\n", res);
            return 1;
        }

        CUdevice device;
        res = cuDeviceGet(&device, i);
        if (res != CUDA_SUCCESS) {
            printf("Failed to get device: %d\n", res);
            return 1;
        }

        int smCount, maxThreadsPerSM, maxBlocksPerSM;
        int regsPerSM, sharedMemPerSM, maxGridWidth, maxGridHeight, maxGridDepth;
        int maxBlockWidth, maxBlockHeight, maxBlockDepth;
        int maxThreadsPerBlock, warpSize, maxRegsPerBlock, maxSharedMemPerBlock;

        res = cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
        res = cuDeviceGetAttribute(&maxThreadsPerSM, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
        res = cuDeviceGetAttribute(&maxBlocksPerSM, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, device);
        res = cuDeviceGetAttribute(&regsPerSM, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device);
        res = cuDeviceGetAttribute(&sharedMemPerSM, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, device);
        res = cuDeviceGetAttribute(&maxGridWidth, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
        res = cuDeviceGetAttribute(&maxGridHeight, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device);
        res = cuDeviceGetAttribute(&maxGridDepth, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device);
        res = cuDeviceGetAttribute(&maxBlockWidth, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
        res = cuDeviceGetAttribute(&maxBlockHeight, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device);
        res = cuDeviceGetAttribute(&maxBlockDepth, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device);
        res = cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
        res = cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
        res = cuDeviceGetAttribute(&maxRegsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device);
        res = cuDeviceGetAttribute(&maxSharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);

        printf("  Max Threads per SM: %d\n", maxThreadsPerSM);
        printf("  Max Blocks per SM: %d\n", maxBlocksPerSM);
        printf("  Registers per SM: %d\n", regsPerSM);
        printf("  Shared Memory per SM (KB): %d\n", sharedMemPerSM / 1024);
        printf("  Max Grid Size: %d x %d x %d\n", maxGridWidth, maxGridHeight, maxGridDepth);
        printf("  Max Block Size: %d x %d x %d\n", maxBlockWidth, maxBlockHeight, maxBlockDepth);
        printf("  Max Threads per Block: %d\n", maxThreadsPerBlock);
        printf("  Warp Size: %d\n", warpSize);
        printf("  Max Registers per Block: %d\n", maxRegsPerBlock);
        printf("  Max Shared Memory per Block (bytes): %d\n", maxSharedMemPerBlock);
    }
    
    return 0;
}

