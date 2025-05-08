#include <vector>
#include <limits>
#include <epi.h>
#include <algorithm>
#include <chrono>
#include <iostream>

// Constants for memory optimization
constexpr size_t kCacheLineSize = 64;
constexpr size_t kPrefetchDistance = 2;
constexpr size_t kStrideSize = 8;  // Optimal stride for gather/scatter
constexpr size_t kBlockSize = 4;   // Number of features to process in a block
constexpr size_t kPrefetchAhead = 3; // Number of cache lines to prefetch ahead

// Performance monitoring structures
struct GatherScatterStats {
    size_t gather_operations = 0;
    size_t scatter_operations = 0;
    size_t cache_misses = 0;
    size_t memory_bandwidth = 0;
    double gather_latency = 0.0;
    double scatter_latency = 0.0;
    double execution_time = 0.0;
    std::chrono::high_resolution_clock::time_point start_time;
};

struct PerformanceMetrics {
    double gather_latency = 0.0;
    double scatter_latency = 0.0;
    size_t memory_bandwidth = 0;
    size_t cache_misses = 0;
    size_t operation_count = 0;
    double vector_utilization = 0.0;
    double cache_hit_rate = 0.0;
    double memory_bandwidth_util = 0.0;
};

struct TreeNode {
    bool is_leaf;
    bool is_categorical;
    size_t feature_index;
    float threshold;
    int32_t left_child;
    int32_t right_child;
    float leaf_value;
    std::vector<size_t> categories;
    // Add stride-based indices for efficient traversal
    alignas(kCacheLineSize) int traversal_indices[kStrideSize];
    // Add prefetch hints
    alignas(kCacheLineSize) uint64_t prefetch_hints[kStrideSize];
};

struct Tree {
    std::vector<TreeNode> nodes;
    // Cache-aware data structure
    alignas(kCacheLineSize) std::vector<float> thresholds_cache;
    alignas(kCacheLineSize) std::vector<int32_t> children_cache;
    alignas(kCacheLineSize) std::vector<float> leaf_values_cache;
};

class XGBoostPredictor {
private:
    static constexpr size_t kAlignment = 64;
    static constexpr float kMissingValue = std::numeric_limits<float>::quiet_NaN();

    // Pre-allocated buffers with stride-based access
    alignas(kAlignment) std::vector<float> feature_buffer_;
    alignas(kAlignment) std::vector<float> threshold_buffer_;
    alignas(kAlignment) std::vector<int32_t> left_child_buffer_;
    alignas(kAlignment) std::vector<int32_t> right_child_buffer_;
    alignas(kAlignment) std::vector<size_t> current_nodes_;
    alignas(kAlignment) std::vector<float> tree_predictions_;
    // Stride indices for gather/scatter operations
    alignas(kAlignment) std::vector<int> stride_indices_;
    // Prefetch buffer
    alignas(kAlignment) std::vector<float> prefetch_buffer_;
    // Performance monitoring
    std::unique_ptr<GatherScatterStats> stats_;
    PerformanceMetrics metrics_;

public:
    XGBoostPredictor(size_t max_samples) {
        // Pre-allocate buffers
        feature_buffer_.resize(max_samples);
        threshold_buffer_.resize(max_samples);
        left_child_buffer_.resize(max_samples);
        right_child_buffer_.resize(max_samples);
        current_nodes_.resize(max_samples);
        tree_predictions_.resize(max_samples);
        stride_indices_.resize(kStrideSize);
        prefetch_buffer_.resize(kPrefetchAhead * kCacheLineSize);
        stats_ = std::make_unique<GatherScatterStats>();
        
        // Initialize stride indices for optimal gather/scatter
        for (size_t i = 0; i < kStrideSize; ++i) {
            stride_indices_[i] = i * (max_samples / kStrideSize);
        }
    }

    // Prefetch helper function
    void PrefetchData(const void* addr, size_t size) {
        const char* ptr = static_cast<const char*>(addr);
        for (size_t i = 0; i < size; i += kCacheLineSize) {
            __builtin_prefetch(ptr + i, 0, kPrefetchDistance);
        }
    }

    // Performance monitoring functions
    void ResetPerformanceCounters() {
        stats_->gather_operations = 0;
        stats_->scatter_operations = 0;
        stats_->cache_misses = 0;
        stats_->memory_bandwidth = 0;
        stats_->gather_latency = 0.0;
        stats_->scatter_latency = 0.0;
        stats_->execution_time = 0.0;
        stats_->start_time = std::chrono::high_resolution_clock::now();
    }

    void UpdatePerformanceMetrics() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - stats_->start_time).count();
        
        metrics_.gather_latency = stats_->gather_latency / stats_->gather_operations;
        metrics_.scatter_latency = stats_->scatter_latency / stats_->scatter_operations;
        metrics_.memory_bandwidth = (stats_->gather_operations + stats_->scatter_operations) * 
                                  kCacheLineSize * 1000000000 / duration;
        metrics_.cache_misses = stats_->cache_misses;
        metrics_.operation_count = stats_->gather_operations + stats_->scatter_operations;
        metrics_.vector_utilization = static_cast<double>(stats_->gather_operations) / 
                                    (stats_->gather_operations + stats_->scatter_operations);
        metrics_.cache_hit_rate = 1.0 - static_cast<double>(stats_->cache_misses) / 
                                (stats_->gather_operations + stats_->scatter_operations);
        metrics_.memory_bandwidth_util = static_cast<double>(metrics_.memory_bandwidth) / 
                                       (kCacheLineSize * 1000000000 / duration);
    }

    void PrintPerformanceStats() {
        std::cout << "\nPerformance Metrics:\n";
        std::cout << "-------------------\n";
        std::cout << "Gather Latency: " << metrics_.gather_latency << " ns\n";
        std::cout << "Scatter Latency: " << metrics_.scatter_latency << " ns\n";
        std::cout << "Memory Bandwidth: " << metrics_.memory_bandwidth << " bytes/s\n";
        std::cout << "Cache Misses: " << metrics_.cache_misses << "\n";
        std::cout << "Total Operations: " << metrics_.operation_count << "\n";
        std::cout << "Vector Utilization: " << metrics_.vector_utilization * 100 << "%\n";
        std::cout << "Cache Hit Rate: " << metrics_.cache_hit_rate * 100 << "%\n";
        std::cout << "Memory Bandwidth Utilization: " << metrics_.memory_bandwidth_util * 100 << "%\n";
    }

    void PredictBatch(const std::vector<Tree>& trees,
                     const std::vector<std::vector<float>>& data,
                     std::vector<float>& out_preds) {
        ResetPerformanceCounters();
        
        size_t num_samples = data.size();
        if (num_samples == 0) return;

        out_preds.resize(num_samples);
        std::fill(out_preds.begin(), out_preds.end(), 0.0f);

        unsigned long int gvl = __builtin_epi_vsetvl(num_samples, __epi_e32, __epi_m1);
        
        // Load stride indices for gather/scatter
        __epi_32xi32 stride_vec = __builtin_epi_vload_32xi32(stride_indices_.data(), gvl);

        for (const auto& tree : trees) {
            // Prefetch tree data
            PrefetchData(tree.thresholds_cache.data(), tree.thresholds_cache.size() * sizeof(float));
            PrefetchData(tree.children_cache.data(), tree.children_cache.size() * sizeof(int32_t));
            PrefetchData(tree.leaf_values_cache.data(), tree.leaf_values_cache.size() * sizeof(float));

            PredictSingleTreeWithStride(tree, data, tree_predictions_, stride_vec);
            
            // Accumulate predictions using stride-based scatter with prefetching
            for (size_t i = 0; i < num_samples; i += gvl) {
                size_t current_batch_size = std::min(gvl, num_samples - i);
                
                // Prefetch next batch
                if (i + gvl < num_samples) {
                    PrefetchData(&out_preds[i + gvl], gvl * sizeof(float));
                    PrefetchData(&tree_predictions_[i + gvl], gvl * sizeof(float));
                }
                
                // Load current predictions with stride
                auto gather_start = std::chrono::high_resolution_clock::now();
                __epi_32xf32 preds = __builtin_epi_vgather_32xf32_32xi32(
                    &out_preds[0], stride_vec, current_batch_size);
                stats_->gather_operations++;
                stats_->gather_latency += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - gather_start).count();
                
                // Load tree predictions with stride
                gather_start = std::chrono::high_resolution_clock::now();
                __epi_32xf32 tree_preds = __builtin_epi_vgather_32xf32_32xi32(
                    &tree_predictions_[0], stride_vec, current_batch_size);
                stats_->gather_operations++;
                stats_->gather_latency += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - gather_start).count();
                
                // Add and scatter back with stride
                __epi_32xf32 new_preds = __builtin_epi_vfadd_32xf32_32xf32(
                    preds, tree_preds, current_batch_size);
                
                auto scatter_start = std::chrono::high_resolution_clock::now();
                __builtin_epi_vscatter_32xf32_32xi32(
                    &out_preds[0], stride_vec, new_preds, current_batch_size);
                stats_->scatter_operations++;
                stats_->scatter_latency += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - scatter_start).count();
            }
        }

        UpdatePerformanceMetrics();
        PrintPerformanceStats();
    }

private:
    void PredictSingleTreeWithStride(const Tree& tree,
                                   const std::vector<std::vector<float>>& data,
                                   std::vector<float>& out_preds,
                                   __epi_32xi32 stride_vec) {
        size_t num_samples = data.size();
        if (num_samples == 0) return;

        std::fill(current_nodes_.begin(), current_nodes_.begin() + num_samples, 0);
        std::fill(out_preds.begin(), out_preds.begin() + num_samples, 0.0f);

        unsigned long int gvl = __builtin_epi_vsetvl(num_samples, __epi_e32, __epi_m1);
        
        // Process samples in blocks with stride and prefetching
        for (size_t i = 0; i < num_samples; i += gvl) {
            size_t current_batch_size = std::min(gvl, num_samples - i);
            
            // Prefetch next block of features
            if (i + gvl < num_samples) {
                for (size_t block = 0; block < kBlockSize; ++block) {
                    PrefetchData(&data[i + gvl][block], gvl * sizeof(float));
                }
            }
            
            // Gather features with stride
            std::array<__epi_32xf32, kBlockSize> feature_regs;
            for (size_t block = 0; block < kBlockSize; ++block) {
                auto gather_start = std::chrono::high_resolution_clock::now();
                feature_regs[block] = __builtin_epi_vgather_32xf32_32xi32(
                    &data[i][block], stride_vec, current_batch_size);
                stats_->gather_operations++;
                stats_->gather_latency += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - gather_start).count();
            }
            
            // Initialize current nodes with stride
            auto gather_start = std::chrono::high_resolution_clock::now();
            __epi_32xi32 current_nodes_vec = __builtin_epi_vgather_32xi32_32xi32(
                &current_nodes_[0], stride_vec, current_batch_size);
            stats_->gather_operations++;
            stats_->gather_latency += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - gather_start).count();
            
            // Tree traversal with stride-based processing and prefetching
            while (true) {
                bool all_leaf = true;
                
                // Process each block of features with stride
                for (size_t block = 0; block < kBlockSize; ++block) {
                    // Prefetch next level of tree
                    if (block + 1 < kBlockSize) {
                        PrefetchData(&tree.thresholds_cache[block + 1], kStrideSize * sizeof(float));
                        PrefetchData(&tree.children_cache[block + 1], kStrideSize * sizeof(int32_t));
                    }
                    
                    // Gather thresholds with stride
                    gather_start = std::chrono::high_resolution_clock::now();
                    __epi_32xf32 threshold_vec = __builtin_epi_vgather_32xf32_32xi32(
                        &tree.thresholds_cache[block], current_nodes_vec, current_batch_size);
                    stats_->gather_operations++;
                    stats_->gather_latency += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now() - gather_start).count();
                    
                    // Compare features with thresholds
                    __epi_32xi1 mask = __builtin_epi_vflt_32xf32_32xf32(
                        feature_regs[block], threshold_vec, current_batch_size);
                    
                    // Gather child indices with stride
                    gather_start = std::chrono::high_resolution_clock::now();
                    __epi_32xi32 left_vec = __builtin_epi_vgather_32xi32_32xi32(
                        &tree.children_cache[block * 2], current_nodes_vec, current_batch_size);
                    __epi_32xi32 right_vec = __builtin_epi_vgather_32xi32_32xi32(
                        &tree.children_cache[block * 2 + 1], current_nodes_vec, current_batch_size);
                    stats_->gather_operations += 2;
                    stats_->gather_latency += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now() - gather_start).count();
                    
                    current_nodes_vec = __builtin_epi_vmerge_32xi32_32xi1(
                        mask, left_vec, right_vec, current_batch_size);
                    
                    // Check if all nodes are leaves
                    for (size_t j = 0; j < current_batch_size; j += kStrideSize) {
                        if (!tree.nodes[current_nodes_[j]].is_leaf) {
                            all_leaf = false;
                            break;
                        }
                    }
                    
                    if (all_leaf) break;
                }
                
                if (all_leaf) break;
            }
            
            // Prefetch leaf values
            PrefetchData(&tree.leaf_values_cache[0], kStrideSize * sizeof(float));
            
            // Gather leaf values with stride
            gather_start = std::chrono::high_resolution_clock::now();
            __epi_32xf32 leaf_vec = __builtin_epi_vgather_32xf32_32xi32(
                &tree.leaf_values_cache[0], current_nodes_vec, current_batch_size);
            stats_->gather_operations++;
            stats_->gather_latency += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - gather_start).count();
            
            // Scatter leaf values to predictions with stride
            auto scatter_start = std::chrono::high_resolution_clock::now();
            __builtin_epi_vscatter_32xf32_32xi32(
                &out_preds[0], stride_vec, leaf_vec, current_batch_size);
            stats_->scatter_operations++;
            stats_->scatter_latency += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - scatter_start).count();
        }
    }
};

// Example usage:
int main() {
    XGBoostPredictor predictor(1000);
    std::vector<Tree> trees = {/* ... initialize trees ... */};
    std::vector<std::vector<float>> data = {/* ... input data ... */};
    std::vector<float> predictions;
    predictor.PredictBatch(trees, data, predictions);
    return 0;
} 