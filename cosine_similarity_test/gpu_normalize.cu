/*
 * GPU L2 归一化 - 直接调用 Faiss 的 L2Norm Kernel
 */

#include "gpu_normalize.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// 引入 Faiss 类型定义和 GPU 实现的头文件
#include <faiss/Index.h>  // idx_t 类型定义
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/utils/Tensor.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>

// 前向声明 Faiss 的 kernel
namespace faiss {
namespace gpu {

// L2 Norm kernel（来自 faiss/gpu/impl/L2Norm.cu）
template <typename T, typename TVec, int RowTileSize, bool NormSquared>
__global__ void l2NormRowMajor(
    Tensor<TVec, 2, true> input,
    Tensor<float, 1, true> output);

// Top-K 选择（来自 faiss/gpu/utils/BlockSelectFloat.cu）
void runBlockSelect(
    Tensor<float, 2, true>& in,
    Tensor<float, 2, true>& outK,
    Tensor<idx_t, 2, true>& outV,
    bool dir,
    int k,
    cudaStream_t stream);

} // namespace gpu
} // namespace faiss

// 简单的归一化 kernel：将每个向量除以其 L2 范数
__global__ void normalizeByNormKernel(
    float* vectors,        // [n × d]
    const float* norms,    // [n] - L2 范数
    int n,                 // 向量数量
    int d                  // 向量维度
) {
    int vec_idx = blockIdx.x;  // 向量索引
    int dim_idx = threadIdx.x; // 维度索引
    
    if (vec_idx < n) {
        float norm = norms[vec_idx];
        
        // 避免除以 0
        if (norm > 1e-10f) {
            // 每个线程处理多个维度
            for (int i = dim_idx; i < d; i += blockDim.x) {
                vectors[vec_idx * d + i] /= norm;
            }
        }
    }
}

// C++ 接口：GPU L2 归一化（直接调用 Faiss kernel）
extern "C" void gpuNormalizeL2(
    float* d_vectors,      // GPU 内存中的向量
    int n,                 // 向量数量
    int d,                 // 向量维度
    cudaStream_t stream
) {
    using namespace faiss::gpu;
    
    // 1. 分配临时内存存储范数
    float* d_norms;
    cudaMalloc(&d_norms, n * sizeof(float));
    
    // 2. 创建 Faiss Tensor 封装
    // input: [n, d] 行主序
    Tensor<float, 2, true> inputTensor(d_vectors, {(int)n, (int)d});
    // output: [n] 范数
    Tensor<float, 1, true> outputTensor(d_norms, {(int)n});
    
    // 3. 直接调用 Faiss 的 l2NormRowMajor kernel
    // 参数设置参考 faiss/gpu/impl/L2Norm.cu 的 runL2Norm 函数
    constexpr int RowTileSize = 8;  // Faiss 使用的 tile 大小
    int warpSize = 32;  // 标准 warp 大小
    
    // 检查是否可以使用 float4 向量化加载
    bool canUseFloat4 = (d % 4 == 0) && (reinterpret_cast<uintptr_t>(d_vectors) % 16 == 0);
    
    if (canUseFloat4) {
        // 使用 float4 向量化加载
        auto inputV = inputTensor.template castResize<float4>();
        auto dim = inputV.getSize(1);
        
        auto numThreads = std::min(
            (int)faiss::gpu::utils::roundUp(dim, warpSize), 
            (int)faiss::gpu::getMaxThreadsCurrentDevice()
        );
        
        auto grid = dim3(faiss::gpu::utils::divUp(inputV.getSize(0), RowTileSize));
        auto block = dim3(numThreads);
        auto smem = sizeof(float) * RowTileSize * faiss::gpu::utils::divUp(numThreads, warpSize);
        
        // 调用 Faiss 的 __global__ kernel，NormSquared=false 表示计算开方后的范数
        faiss::gpu::l2NormRowMajor<float, float4, RowTileSize, false>
            <<<grid, block, smem, stream>>>(inputV, outputTensor);
    } else {
        // 不使用向量化
        auto dim = inputTensor.getSize(1);
        
        auto numThreads = std::min(
            (int)faiss::gpu::utils::roundUp(dim, warpSize),
            (int)faiss::gpu::getMaxThreadsCurrentDevice()
        );
        
        auto grid = dim3(faiss::gpu::utils::divUp(inputTensor.getSize(0), RowTileSize));
        auto block = dim3(numThreads);
        auto smem = sizeof(float) * RowTileSize * faiss::gpu::utils::divUp(numThreads, warpSize);
        
        // 调用 Faiss 的 __global__ kernel
        faiss::gpu::l2NormRowMajor<float, float, RowTileSize, false>
            <<<grid, block, smem, stream>>>(inputTensor, outputTensor);
    }
    
    // 4. 归一化向量（除以范数）
    int threads = std::min(256, d);
    int blocks = n;
    
    normalizeByNormKernel<<<blocks, threads, 0, stream>>>(
        d_vectors, d_norms, n, d
    );
    
    // 5. 清理
    cudaFree(d_norms);
}

// 验证归一化（计算第一个向量的范数）
extern "C" float verifyNormalization(
    const float* d_vectors,
    int d,
    cudaStream_t stream
) {
    using namespace faiss::gpu;
    
    float* d_norm;
    cudaMalloc(&d_norm, sizeof(float));
    
    // 创建 Tensor
    Tensor<float, 2, true> inputTensor(const_cast<float*>(d_vectors), {1, d});
    Tensor<float, 1, true> outputTensor(d_norm, {1});
    
    // 调用 Faiss kernel
    constexpr int RowTileSize = 8;
    int warpSize = 32;
    auto dim = inputTensor.getSize(1);
    auto numThreads = std::min(
        (int)faiss::gpu::utils::roundUp(dim, warpSize),
        (int)faiss::gpu::getMaxThreadsCurrentDevice()
    );
    
    auto grid = dim3(1);
    auto block = dim3(numThreads);
    auto smem = sizeof(float) * RowTileSize * faiss::gpu::utils::divUp(numThreads, warpSize);
    
    faiss::gpu::l2NormRowMajor<float, float, RowTileSize, false>
        <<<grid, block, smem, stream>>>(inputTensor, outputTensor);
    
    float h_norm;
    cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_norm);
    
    return h_norm;
}

// GPU Top-K 选择（直接调用 Faiss 的 blockSelect kernel）
extern "C" void gpuTopKSelect(
    float* d_distances,         // [nq × nb] 距离矩阵（GPU）
    float* d_topk_distances,    // [nq × k] 输出距离（GPU）
    int64_t* d_topk_indices,    // [nq × k] 输出索引（GPU）
    int nq,                     // 查询数量
    int nb,                     // 数据库大小
    int k,                      // Top-K
    cudaStream_t stream
) {
    using namespace faiss::gpu;
    using faiss::idx_t;  // 引入 idx_t 类型（定义在 faiss/MetricType.h）
    
    // 创建 Faiss Tensor 封装
    // in: [nq, nb] 距离矩阵
    Tensor<float, 2, true> inTensor(d_distances, {nq, nb});
    // outK: [nq, k] Top-K 距离
    Tensor<float, 2, true> outKTensor(d_topk_distances, {nq, k});
    // outV: [nq, k] Top-K 索引
    Tensor<idx_t, 2, true> outVTensor(d_topk_indices, {nq, k});
    
    // 调用 Faiss 的 runBlockSelect
    // dir = true 表示选择最大值（余弦相似度越大越好）
    faiss::gpu::runBlockSelect(
        inTensor,
        outKTensor,
        outVTensor,
        true,  // dir: true = 选择最大值
        k,
        stream
    );
}
