/*
 * 独立的 Top-K Kernel 头文件
 */

#ifndef TOPK_KERNEL_H
#define TOPK_KERNEL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * GPU Top-K 选择（独立实现，无 Faiss 依赖）
 * 
 * 参数：
 *   d_input: [num_rows × num_cols] 输入矩阵（GPU）
 *   d_outK: [num_rows × k] 输出 Top-K 值（GPU）
 *   d_outV: [num_rows × k] 输出 Top-K 索引（GPU）
 *   num_rows: 行数
 *   num_cols: 列数
 *   k: Top-K 数量（必须 ≤ 2048）
 *   select_max: true=选择最大值, false=选择最小值
 *   stream: CUDA stream
 */
void launchTopKKernel(
    const float* d_input,
    float* d_outK,
    int* d_outV,
    int num_rows,
    int num_cols,
    int k,
    bool select_max,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // TOPK_KERNEL_H

