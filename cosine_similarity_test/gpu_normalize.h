/*
 * GPU L2 归一化接口
 */

#ifndef GPU_NORMALIZE_H
#define GPU_NORMALIZE_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPU L2 归一化
void gpuNormalizeL2(
    float* d_vectors,      // GPU 内存中的向量
    int n,                 // 向量数量
    int d,                 // 向量维度
    cudaStream_t stream
);

// 验证归一化
float verifyNormalization(
    const float* d_vectors,
    int d,
    cudaStream_t stream
);

// GPU Top-K 选择（直接调用 Faiss 的 blockSelect kernel）
void gpuTopKSelect(
    float* d_distances,         // [nq × nb] 距离矩阵（GPU）
    float* d_topk_distances,    // [nq × k] 输出距离（GPU）
    int64_t* d_topk_indices,    // [nq × k] 输出索引（GPU）
    int nq,                     // 查询数量
    int nb,                     // 数据库大小
    int k,                      // Top-K
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // GPU_NORMALIZE_H

