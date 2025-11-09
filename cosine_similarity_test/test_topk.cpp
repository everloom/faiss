/*
 * Top-K Kernel 测试程序
 * 测试独立的 Top-K kernel 实现
 */

#include <cuda_runtime.h>
#include "topk_kernel.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <chrono>

// CUDA 错误检查
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// 生成随机矩阵
void generateRandomMatrix(float* data, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(42);  // 固定种子
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < rows * cols; i++) {
        data[i] = dis(gen);
    }
}

// CPU 参考实现：选择每行的 Top-K
void cpuTopK(
    const float* input,
    float* outK,
    int* outV,
    int num_rows,
    int num_cols,
    int k
) {
    for (int row = 0; row < num_rows; row++) {
        const float* row_data = input + row * num_cols;
        
        // 创建索引-值对
        std::vector<std::pair<float, int>> pairs;
        for (int col = 0; col < num_cols; col++) {
            pairs.push_back({row_data[col], col});
        }
        
        // 部分排序，选择最大的 k 个
        std::partial_sort(
            pairs.begin(),
            pairs.begin() + k,
            pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
        
        // 提取结果
        for (int i = 0; i < k; i++) {
            outK[row * k + i] = pairs[i].first;
            outV[row * k + i] = pairs[i].second;
        }
    }
}

// 验证结果
bool verifyResults(
    const float* gpu_values,
    const int* gpu_indices,
    const float* cpu_values,
    const int* cpu_indices,
    int num_rows,
    int k
) {
    float max_error = 0.0f;
    int num_errors = 0;
    const float tolerance = 1e-4f;
    
    std::cout << "\n验证结果:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    // 验证前几行
    int verify_rows = std::min(3, num_rows);
    int verify_k = std::min(5, k);
    
    for (int row = 0; row < verify_rows; row++) {
        std::cout << "\n行 " << row << " 的 Top-" << verify_k << ":" << std::endl;
        
        for (int i = 0; i < verify_k; i++) {
            int idx = row * k + i;
            float gpu_val = gpu_values[idx];
            int gpu_idx = gpu_indices[idx];
            float cpu_val = cpu_values[idx];
            int cpu_idx = cpu_indices[idx];
            
            float error = std::abs(gpu_val - cpu_val);
            max_error = std::max(max_error, error);
            
            if (error > tolerance || gpu_idx != cpu_idx) {
                num_errors++;
            }
            
            std::cout << "  Rank " << (i+1) << ": "
                      << "GPU [idx=" << std::setw(6) << gpu_idx 
                      << ", val=" << std::setw(10) << std::fixed << std::setprecision(6) << gpu_val << "] | "
                      << "CPU [idx=" << std::setw(6) << cpu_idx 
                      << ", val=" << std::setw(10) << std::fixed << std::setprecision(6) << cpu_val << "]"
                      << " | Error: " << std::scientific << error
                      << (error > tolerance ? " ⚠️" : " ✓")
                      << std::endl;
        }
    }
    
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "验证统计:" << std::endl;
    std::cout << "  最大误差: " << std::scientific << max_error << std::endl;
    std::cout << "  错误数量: " << num_errors << " / " << (verify_rows * verify_k) << std::endl;
    std::cout << "  结果: " << (num_errors == 0 ? "✓ 通过" : "✗ 失败") << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    return num_errors == 0;
}

// 性能测试
void performanceBenchmark(
    float* d_input,
    float* d_outK,
    int* d_outV,
    int num_rows,
    int num_cols,
    int k
) {
    std::cout << "\n性能测试（运行 100 次）:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    const int num_iterations = 100;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // 预热
    launchTopKKernel(d_input, d_outK, d_outV, num_rows, num_cols, k, true, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        launchTopKKernel(d_input, d_outK, d_outV, num_rows, num_cols, k, true, stream);
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double avg_time = total_time / num_iterations;
    
    // 计算吞吐量
    double elements_per_sec = (num_rows * num_cols) / (avg_time / 1000.0);
    double bandwidth_gb_s = (num_rows * num_cols * sizeof(float)) / (avg_time / 1000.0) / 1e9;
    
    std::cout << "  配置: " << num_rows << " × " << num_cols << " → Top-" << k << std::endl;
    std::cout << "  平均耗时: " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
    std::cout << "  吞吐量: " << std::fixed << std::setprecision(2) << elements_per_sec / 1e6 << " M elements/s" << std::endl;
    std::cout << "  有效带宽: " << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main() {
    std::cout << "\n";
    std::cout << "===============================================================================\n";
    std::cout << "                    独立 Top-K Kernel 测试程序\n";
    std::cout << "===============================================================================\n";
    
    // ========================================================================
    // 测试配置
    // ========================================================================
    const int num_rows = 100;      // 行数（例如 query 数量）
    const int num_cols = 10000;    // 列数（例如 database 大小）
    const int k = 10;              // Top-K
    
    std::cout << "\n测试配置:" << std::endl;
    std::cout << "  输入矩阵: " << num_rows << " × " << num_cols << std::endl;
    std::cout << "  Top-K: " << k << std::endl;
    std::cout << "  选择: 最大值" << std::endl;
    std::cout << "===============================================================================\n" << std::endl;
    
    // ========================================================================
    // 分配内存
    // ========================================================================
    std::cout << "[步骤 1] 分配内存..." << std::endl;
    
    // CPU 内存
    std::vector<float> h_input(num_rows * num_cols);
    std::vector<float> h_outK_gpu(num_rows * k);
    std::vector<int> h_outV_gpu(num_rows * k);
    std::vector<float> h_outK_cpu(num_rows * k);
    std::vector<int> h_outV_cpu(num_rows * k);
    
    // GPU 内存
    float *d_input, *d_outK;
    int *d_outV;
    
    CUDA_CHECK(cudaMalloc(&d_input, num_rows * num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outK, num_rows * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outV, num_rows * k * sizeof(int)));
    
    std::cout << "  ✓ CPU 内存: " << (num_rows * num_cols * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  ✓ GPU 内存: " << (num_rows * num_cols * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // ========================================================================
    // 生成测试数据
    // ========================================================================
    std::cout << "\n[步骤 2] 生成随机测试数据..." << std::endl;
    
    generateRandomMatrix(h_input.data(), num_rows, num_cols);
    
    std::cout << "  ✓ 生成 " << num_rows << " × " << num_cols << " 随机矩阵" << std::endl;
    std::cout << "  数据范围示例: [" << h_input[0] << ", " << h_input[1] << ", ...]" << std::endl;
    
    // 拷贝到 GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), 
                          num_rows * num_cols * sizeof(float), 
                          cudaMemcpyHostToDevice));
    
    // ========================================================================
    // GPU Top-K 计算
    // ========================================================================
    std::cout << "\n[步骤 3] GPU Top-K 计算..." << std::endl;
    std::cout << "  调用: launchTopKKernel()" << std::endl;
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    launchTopKKernel(
        d_input, d_outK, d_outV,
        num_rows, num_cols, k,
        true,  // 选择最大值
        stream
    );
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    double gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    std::cout << "  ✓ GPU 计算完成 (耗时: " << std::fixed << std::setprecision(3) << gpu_time << " ms)" << std::endl;
    
    // 拷贝结果回 CPU
    CUDA_CHECK(cudaMemcpy(h_outK_gpu.data(), d_outK, 
                          num_rows * k * sizeof(float), 
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_outV_gpu.data(), d_outV, 
                          num_rows * k * sizeof(int), 
                          cudaMemcpyDeviceToHost));
    
    // ========================================================================
    // CPU Top-K 计算（参考结果）
    // ========================================================================
    std::cout << "\n[步骤 4] CPU Top-K 计算（用于验证）..." << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    cpuTopK(h_input.data(), h_outK_cpu.data(), h_outV_cpu.data(), 
            num_rows, num_cols, k);
    end = std::chrono::high_resolution_clock::now();
    
    double cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    std::cout << "  ✓ CPU 计算完成 (耗时: " << std::fixed << std::setprecision(3) << cpu_time << " ms)" << std::endl;
    std::cout << "  加速比: " << std::fixed << std::setprecision(2) << cpu_time / gpu_time << "×" << std::endl;
    
    // ========================================================================
    // 验证结果
    // ========================================================================
    bool passed = verifyResults(
        h_outK_gpu.data(), h_outV_gpu.data(),
        h_outK_cpu.data(), h_outV_cpu.data(),
        num_rows, k
    );
    
    // ========================================================================
    // 性能测试
    // ========================================================================
    performanceBenchmark(d_input, d_outK, d_outV, num_rows, num_cols, k);
    
    // ========================================================================
    // 多配置测试
    // ========================================================================
    std::cout << "\n[步骤 5] 多配置性能测试..." << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    std::cout << std::setw(10) << "Rows"
              << " | " << std::setw(10) << "Cols"
              << " | " << std::setw(8) << "TopK"
              << " | " << std::setw(12) << "Time(ms)"
              << " | " << std::setw(12) << "Throughput"
              << " | " << std::setw(12) << "Bandwidth"
              << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    struct TestConfig {
        int rows, cols, k;
    };
    
    std::vector<TestConfig> configs = {
        {1, 10000, 10},
        {8, 30000, 256},
        {8, 60000, 512},
        {8, 60000, 1024},
        {8, 60000, 2048},
        {100, 10000, 10},
        {100, 10000, 100},
        {100, 50000, 1000},
    };
    
    for (const auto& cfg : configs) {
        // 重新分配内存（如果需要）
        if (cfg.rows * cfg.cols > num_rows * num_cols || cfg.k > k) {
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaFree(d_outK));
            CUDA_CHECK(cudaFree(d_outV));
            
            CUDA_CHECK(cudaMalloc(&d_input, cfg.rows * cfg.cols * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_outK, cfg.rows * cfg.k * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_outV, cfg.rows * cfg.k * sizeof(int)));
        }
        
        // 预热
        launchTopKKernel(d_input, d_outK, d_outV, cfg.rows, cfg.cols, cfg.k, true, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // 测试 10 次取平均
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; i++) {
            launchTopKKernel(d_input, d_outK, d_outV, cfg.rows, cfg.cols, cfg.k, true, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        end = std::chrono::high_resolution_clock::now();
        
        double avg_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 10000.0;
        double throughput = (cfg.rows * cfg.cols) / (avg_time / 1000.0) / 1e6;
        double bandwidth = (cfg.rows * cfg.cols * sizeof(float)) / (avg_time / 1000.0) / 1e9;
        
        std::cout << std::setw(10) << cfg.rows
                  << " | " << std::setw(10) << cfg.cols
                  << " | " << std::setw(8) << cfg.k
                  << " | " << std::setw(12) << std::fixed << std::setprecision(3) << avg_time
                  << " | " << std::setw(12) << std::fixed << std::setprecision(2) << throughput << " M/s"
                  << " | " << std::setw(12) << std::fixed << std::setprecision(2) << bandwidth << " GB/s"
                  << std::endl;
    }
    
    std::cout << std::string(100, '=') << std::endl;
    
    // ========================================================================
    // 清理
    // ========================================================================
    std::cout << "\n[步骤 6] 清理资源..." << std::endl;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_outK));
    CUDA_CHECK(cudaFree(d_outV));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    std::cout << "  ✓ GPU 内存已释放" << std::endl;
    
    std::cout << "\n===============================================================================\n";
    if (passed) {
        std::cout << "                         ✓ 测试通过！\n";
    } else {
        std::cout << "                         ✗ 测试失败！\n";
    }
    std::cout << "===============================================================================\n\n";
    
    return passed ? 0 : 1;
}

