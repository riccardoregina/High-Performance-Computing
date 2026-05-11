# High Performance Computing with CUDA

This project demonstrates various CUDA implementations, such as numerical integration using the trapezoidal rule. 

---

## File Overview

### trapezio/
In this directory, we implement and compare multiple strategies for parallel reduction, ranging from a sequential CPU baseline to optimized multi-warp GPU kernels. Each implementation is benchmarked across multiple block sizes to evaluate performance and numerical accuracy.

| File | Description |
|------|-------------|
| `main.cu` | Driver program. Runs all 7 integration algorithms with varying block sizes (64 to 256), collects timing and error metrics. |
| `trapezio.cu` | Implementation of the mathematical function `f(x) = x * exp(-x) * cos(2x)` and all 7 integration kernels: sequential CPU, atomic global, shared memory tree, dissemination, warp shuffle, multi-warp shared memory, and multi-warp shuffle. |
| `trapezio.h` | Header file containing constants (`WARP_SIZE = 32`, `MAX_BLKSZ = 1024`) and function declarations. |
| `utils.cpp` | Utility functions: `get_cur_time()` for high-resolution timing, `float_avg()` and `double_avg()` for computing means across multiple runs. |
| `utils.h` | Header for utility functions. |
| `Makefile` | Build system for compiling the project. |

### utils/

| File | Description |
|------|-------------|
| `device.cu` | Utility that queries and displays CUDA device properties (device name, compute capability, SM count, global memory, etc.). |
| `occupancy-calculator.py` | Python script that computes theoretical GPU occupancy based on device configuration and launch parameters. Add your GPU object (following the `gtx1060` example) containing the required information  and `launch` dictionaries to match your target hardware and kernel launch settings. Note: you can acquire the number of registers per thread by running your kernel with `nsys`. Concerning the hardware details, you can acquire them by running `device.cu`. |

---

## Building and Running

### Build (Makefile)

Ensure the Makefile targets `main.cu`. Then:

```bash
cd trapezio
make
```

This produces the executable (e.g., `trapezio.out`).

### Build (Manual Compilation)

```bash
cd trapezio
nvcc -O3 main.cu trapezio.cu utils.cpp -o trapezio.out
```

### Run

```bash
./trapezio.out
```

The program computes the integral using all 7 implementations, testing multiple block sizes, and prints timing and error results for each configuration.

### Build & Run device.cu

```bash
cd utils
nvcc device.cu -o device.out -lcuda
./device.out
```

This prints detailed information about all CUDA-capable devices on the system.

### Run occupancy-calculator.py

```bash
cd utils
python occupancy-calculator.py
```

This calculates the theoretical occupancy of the specified kernel launch on the specified hardware. This can be useful if your GPU does not allow complete profiling.

---

## CUDA Requirements and Profiling

- **Minimum compute capability**: 3.0 (Kepler and later)
- **CUDA version**: Check with `nvcc --version`

### NVIDIA Nsight Systems (nsys)

Kernel-level tracing and timeline analysis:

```bash
nsys profile -o report_name --export=sqlite ./trapezio.out
```

This generates a `.sqlite` file containing generic trace data. You can query kernel performance with SQL:

```sql
SELECT gridId,
       (end - start) AS time,
       registersPerThread,
       gridX, gridY, gridZ,
       blockX, blockY, blockZ,
       staticSharedMemory,
       dynamicSharedMemory,
       localMemoryPerThread,
       localMemoryTotal,
       sharedMemoryLimit
FROM CUPTI_ACTIVITY_KIND_KERNEL;
```

This query returns kernel execution time, register/shared memory usage, and grid/block dimensions.

### NVIDIA Nsight Compute (ncu)

Kernel-level profiling for very detailed metrics (achieved occupancy, warp stall reasons, memory access patterns, etc.):

```bash
ncu --set summary ./trapezio.out
```

Use `--set full` for a comprehensive report, or target specific kernels with `--kernel-name <pattern>`.

### Profiling Support Warning

**Not all GPUs support profiling** — some older hardware and certain newer architectures may have limited or no profiling capabilities. Verify your GPU's profiling support before attempting to collect trace data. If profiling is unavailable, use `occupancy-calculator.py` (see above) for theoretical occupancy estimates.

---
