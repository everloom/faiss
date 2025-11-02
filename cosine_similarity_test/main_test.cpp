/*
 * 余弦相似度 GPU Kernel Benchmark - 完整版
 * 直接调用 Faiss 底层 Kernel
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gpu_normalize.h"
 #include <iostream>
 #include <vector>
 #include <cmath>
 #include <random>
 #include <iomanip>
 #include <chrono>
#include <algorithm>
#include <fstream>

// ============================================================================
// Benchmark 配置结构
// ============================================================================
struct BenchmarkConfig {
    int query_bs;
    int db_bs;
    int seq_len;
    int top_k;
    
    double normalize_time_ms;
    double gemm_time_ms;
    double topk_time_ms;
    double total_time_ms;
    double gflops;
};

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// 辅助函数
// ============================================================================

 void generateRandomVectors(float* data, size_t n, size_t d) {
     std::random_device rd;
    std::mt19937 gen(42);
     std::normal_distribution<float> dis(0.0f, 1.0f);
     
     for (size_t i = 0; i < n * d; i++) {
         data[i] = dis(gen);
     }
 }
 
std::vector<BenchmarkConfig> generateBenchmarkConfigs() {
    std::vector<BenchmarkConfig> configs;
    
    std::vector<int> query_batch_sizes = {1, 2, 4, 8};
    std::vector<int> db_batch_sizes = {30000, 40000, 50000, 60000};
    std::vector<int> seq_lengths = {64, 128, 256};
    // Top-K 限制在 2048 以内（Faiss GPU_MAX_SELECTION_K 限制）
    std::vector<int> topk_values = {256, 512, 1024, 1536, 2048};
    
    for (int qbs : query_batch_sizes) {
        for (int dbs : db_batch_sizes) {
            for (int sl : seq_lengths) {
                for (int k : topk_values) {
                    BenchmarkConfig config;
                    config.query_bs = qbs;
                    config.db_bs = dbs;
                    config.seq_len = sl;
                    config.top_k = k;
                    config.normalize_time_ms = 0.0;
                    config.gemm_time_ms = 0.0;
                    config.topk_time_ms = 0.0;
                    config.total_time_ms = 0.0;
                    config.gflops = 0.0;
                    
                    configs.push_back(config);
                }
            }
        }
    }
    
    return configs;
}

// ============================================================================
// 验证函数
// ============================================================================

bool verifyResults(
    const BenchmarkConfig& config,
    float* h_database,
    float* h_queries,
    float* d_topk_distances,
    int64_t* d_topk_indices
) {
    // 拷贝 Top-K 结果回 CPU
    std::vector<float> topk_distances(config.query_bs * config.top_k);
    std::vector<int64_t> topk_indices(config.query_bs * config.top_k);
    
    CUDA_CHECK(cudaMemcpy(topk_distances.data(), d_topk_distances,
                          config.query_bs * config.top_k * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(topk_indices.data(), d_topk_indices,
                          config.query_bs * config.top_k * sizeof(int64_t),
                          cudaMemcpyDeviceToHost));
    
    // 验证前3个查询的前5个结果
     float max_error = 0.0f;
     int num_errors = 0;
     const float tolerance = 1e-4f;
    const int verify_queries = std::min(3, config.query_bs);
    const int verify_k = std::min(5, config.top_k);
    
    for (int i = 0; i < verify_queries; i++) {
        for (int j = 0; j < verify_k; j++) {
            int64_t db_idx = topk_indices[i * config.top_k + j];
            float gpu_similarity = topk_distances[i * config.top_k + j];
            
            // CPU 计算余弦相似度（已归一化，直接内积）
            float cpu_similarity = 0.0f;
            for (int d = 0; d < config.seq_len; d++) {
                cpu_similarity += h_queries[i * config.seq_len + d] * 
                                 h_database[db_idx * config.seq_len + d];
            }
             
             float error = std::abs(gpu_similarity - cpu_similarity);
             max_error = std::max(max_error, error);
             
             if (error > tolerance) {
                 num_errors++;
             }
        }
    }
    
    std::cout << "\n  验证结果 (前" << verify_queries << "个查询，前" << verify_k << "个结果):" << std::endl;
    std::cout << "    最大误差: " << std::scientific << max_error << std::endl;
    std::cout << "    超过阈值数量: " << num_errors << std::endl;
    std::cout << "    状态: " << (num_errors == 0 ? "✓ 通过" : "✗ 失败") << std::endl;
    
    return num_errors == 0;
}

// ============================================================================
// Benchmark 核心函数
// ============================================================================

BenchmarkConfig runSingleBenchmark(
    const BenchmarkConfig& config,
    cublasHandle_t& handle,
    float* h_database,
    float* h_queries,
    float* d_database,
    float* d_queries,
    float* d_distances,
    float* d_topk_distances,
    int64_t* d_topk_indices,
    bool verify = false
) {
    BenchmarkConfig result = config;
    
    // 生成随机数据
    generateRandomVectors(h_database, config.db_bs, config.seq_len);
    generateRandomVectors(h_queries, config.query_bs, config.seq_len);
    
    // 拷贝到 GPU
    CUDA_CHECK(cudaMemcpy(d_database, h_database, 
                          config.db_bs * config.seq_len * sizeof(float), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, 
                          config.query_bs * config.seq_len * sizeof(float), 
                          cudaMemcpyHostToDevice));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    
    // 1. L2 归一化
    start = std::chrono::high_resolution_clock::now();
    gpuNormalizeL2(d_database, config.db_bs, config.seq_len, stream);
    gpuNormalizeL2(d_queries, config.query_bs, config.seq_len, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    end = std::chrono::high_resolution_clock::now();
    result.normalize_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    // 如果需要验证，拷贝归一化后的数据回 CPU
    if (verify) {
        CUDA_CHECK(cudaMemcpy(h_database, d_database,
                              config.db_bs * config.seq_len * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_queries, d_queries,
                              config.query_bs * config.seq_len * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    
    // 2. GEMM 计算
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    start = std::chrono::high_resolution_clock::now();
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        config.db_bs,
        config.query_bs,
        config.seq_len,
        &alpha,
        d_database,
        config.seq_len,
        d_queries,
        config.seq_len,
        &beta,
        d_distances,
        config.db_bs
    ));
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    result.gemm_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    result.gflops = (2.0 * config.query_bs * config.db_bs * config.seq_len) / (result.gemm_time_ms / 1000.0) / 1e9;
    
    // 3. Top-K 选择
    start = std::chrono::high_resolution_clock::now();
    gpuTopKSelect(d_distances, d_topk_distances, d_topk_indices, 
                  config.query_bs, config.db_bs, config.top_k, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    end = std::chrono::high_resolution_clock::now();
    result.topk_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    result.total_time_ms = result.normalize_time_ms + result.gemm_time_ms + result.topk_time_ms;
    
    // 验证结果（仅在第一次运行时）
    if (verify) {
        verifyResults(config, h_database, h_queries, d_topk_distances, d_topk_indices);
    }
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    return result;
}

// ============================================================================
// Main 函数
// ============================================================================

int main() {
    try {
        std::cout << "\n";
        std::cout << "===============================================================================\n";
        std::cout << "   余弦相似度 GPU Kernel Benchmark - 直接调用 Faiss 底层 Kernel\n";
        std::cout << "===============================================================================\n";
        std::cout << "\nBenchmark 配置:\n";
        std::cout << "  查询 Batch Size: 1, 2, 4, 8\n";
        std::cout << "  数据库 Batch Size: 30000, 40000, 50000, 60000\n";
        std::cout << "  Sequence Length: 64, 128, 256\n";
        std::cout << "  Top-K: 256, 512, 1024, 1536, 2048 (Faiss GPU 最大限制)\n";
        std::cout << "===============================================================================\n\n";
        
        // 生成 benchmark 配置
        auto configs = generateBenchmarkConfigs();
        std::cout << "总共生成 " << configs.size() << " 个测试配置\n\n";
        
        // 分配最大可能的 GPU 内存
        const int max_query_bs = 8;
        const int max_db_bs = 60000;
        const int max_seq_len = 256;
        const int max_top_k = 2048;  // Faiss GPU_MAX_SELECTION_K 限制
        
        std::cout << "[步骤 1] 分配 GPU 内存（最大配置）..." << std::endl;
        
        float *d_database, *d_queries, *d_distances;
        float *d_topk_distances;
        int64_t *d_topk_indices;
        
        CUDA_CHECK(cudaMalloc(&d_database, max_db_bs * max_seq_len * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_queries, max_query_bs * max_seq_len * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_distances, max_query_bs * max_db_bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_topk_distances, max_query_bs * max_top_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_topk_indices, max_query_bs * max_top_k * sizeof(int64_t)));
        
        std::cout << "  ✓ GPU 内存分配完成" << std::endl;
        std::cout << "    - Database: " << (max_db_bs * max_seq_len * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "    - Queries: " << (max_query_bs * max_seq_len * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "    - Distances: " << (max_query_bs * max_db_bs * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
        
        // 分配 CPU 内存
        std::vector<float> h_database(max_db_bs * max_seq_len);
        std::vector<float> h_queries(max_query_bs * max_seq_len);
        
        // 创建 cuBLAS handle
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        std::cout << "  ✓ cuBLAS handle 创建完成\n" << std::endl;
        
        // 打开输出文件
        std::ofstream csv_file("benchmark_results.csv");
        csv_file << "QueryBS,DBBS,SeqLen,TopK,NormalizeTime(ms),GEMMTime(ms),TopKTime(ms),TotalTime(ms),GFLOPS\n";
        
        // 运行 benchmark
        std::cout << "[步骤 2] 开始运行 Benchmark...\n" << std::endl;
        std::cout << std::string(120, '-') << std::endl;
        std::cout << std::setw(8) << "QueryBS"
                  << " | " << std::setw(6) << "DBBS"
                  << " | " << std::setw(7) << "SeqLen"
                  << " | " << std::setw(5) << "TopK"
                  << " | " << std::setw(10) << "Norm(ms)"
                  << " | " << std::setw(10) << "GEMM(ms)"
                  << " | " << std::setw(10) << "TopK(ms)"
                  << " | " << std::setw(10) << "Total(ms)"
                  << " | " << std::setw(10) << "GFLOPS"
                  << std::endl;
        std::cout << std::string(120, '-') << std::endl;
        
        int count = 0;
        bool first_run = true;
        
        for (const auto& config : configs) {
            // 第一次运行时验证结果正确性
            bool verify = first_run;
            
            auto result = runSingleBenchmark(
                config, handle,
                h_database.data(), h_queries.data(),
                d_database, d_queries, d_distances,
                d_topk_distances, d_topk_indices,
                verify
            );
            
            if (first_run) {
                first_run = false;
                std::cout << std::string(120, '-') << std::endl;
            }
            
            // 输出到控制台
            std::cout << std::setw(8) << result.query_bs
                      << " | " << std::setw(6) << result.db_bs
                      << " | " << std::setw(7) << result.seq_len
                      << " | " << std::setw(5) << result.top_k
                      << " | " << std::setw(10) << std::fixed << std::setprecision(3) << result.normalize_time_ms
                      << " | " << std::setw(10) << std::fixed << std::setprecision(3) << result.gemm_time_ms
                      << " | " << std::setw(10) << std::fixed << std::setprecision(3) << result.topk_time_ms
                      << " | " << std::setw(10) << std::fixed << std::setprecision(3) << result.total_time_ms
                      << " | " << std::setw(10) << std::fixed << std::setprecision(2) << result.gflops
                      << std::endl;
            
            // 输出到 CSV
            csv_file << result.query_bs << ","
                     << result.db_bs << ","
                     << result.seq_len << ","
                     << result.top_k << ","
                     << result.normalize_time_ms << ","
                     << result.gemm_time_ms << ","
                     << result.topk_time_ms << ","
                     << result.total_time_ms << ","
                     << result.gflops << "\n";
            
            count++;
            if (count % 20 == 0) {
                std::cout << "  进度: " << count << " / " << configs.size() << std::endl;
            }
        }
        
        std::cout << std::string(120, '-') << std::endl;
        std::cout << "\n✓ Benchmark 完成！共测试 " << configs.size() << " 个配置" << std::endl;
        std::cout << "✓ 结果已保存到 benchmark_results.csv\n" << std::endl;
        
        // 清理资源
        csv_file.close();
        CUDA_CHECK(cudaFree(d_database));
        CUDA_CHECK(cudaFree(d_queries));
        CUDA_CHECK(cudaFree(d_distances));
        CUDA_CHECK(cudaFree(d_topk_distances));
        CUDA_CHECK(cudaFree(d_topk_indices));
        CUBLAS_CHECK(cublasDestroy(handle));
        
        std::cout << "✓ GPU 资源已清理" << std::endl;
        std::cout << "\n程序执行成功！\n" << std::endl;
         
         return 0;
         
     } catch (const std::exception& e) {
         std::cerr << "错误: " << e.what() << std::endl;
         return 1;
     }
 }
