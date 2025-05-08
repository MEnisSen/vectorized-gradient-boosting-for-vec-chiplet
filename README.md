# Vectorized Gradient Boosting for EUPilot VEC Chiplet

This repository contains an optimized gradient boosting inference implementation, designed for the EUPilot VEC chiplet, a cutting-edge, low-power, and highly scalable vector processing unit aimed at accelerating machine learning workloads. The code provided leverages both manual vectorization and compiler-assisted optimizations to achieve significant performance gains over scalar implementations.

### **Key Features**
- **Manual Vectorization with Intrinsics:** Optimized tree traversal and prediction logic using RISC-V intrinsics for gather, scatter, and masked operations.
- **Cache-Aware Data Structures:** Aligned memory layouts to reduce cache misses and improve memory bandwidth.
- **Stride-Based Processing:** Efficient data access patterns for block-wise vector processing.
- **Prefetching and Parallelism:** Reduced memory latency through prefetch hints and parallelized feature comparisons.
- **Detailed Performance Monitoring:** Real-time tracking of vector utilization, cache hit rates, and memory bandwidth.
