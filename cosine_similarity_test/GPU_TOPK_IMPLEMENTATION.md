# GPU Top-K 实现说明

本文档说明如何直接调用 Faiss 的 GPU Top-K kernel。

## 实现概述

我们的余弦相似度计算现在**完全在 GPU 上完成**，包括三个核心步骤：

1. **L2 归一化**: 调用 Faiss 的 `l2NormRowMajor` kernel
2. **内积计算**: 调用 cuBLAS 的 `cublasSgemm`
3. **Top-K 选择**: 调用 Faiss 的 `blockSelect` kernel ⭐ **新增**

## 代码架构

### 1. GPU Normalize 模块

#### `gpu_normalize.h`
```cpp
// GPU Top-K 选择函数声明
void gpuTopKSelect(
    float* d_distances,         // [nq × nb] 距离矩阵（GPU）
    float* d_topk_distances,    // [nq × k] 输出距离（GPU）
    int64_t* d_topk_indices,    // [nq × k] 输出索引（GPU）
    int nq,                     // 查询数量
    int nb,                     // 数据库大小
    int k,                      // Top-K
    cudaStream_t stream
);
```

#### `gpu_normalize.cu`
```cpp
// 直接调用 Faiss 的 runBlockSelect
void gpuTopKSelect(...) {
    // 创建 Tensor 封装
    Tensor<float, 2, true> inTensor(d_distances, {nq, nb});
    Tensor<float, 2, true> outKTensor(d_topk_distances, {nq, k});
    Tensor<idx_t, 2, true> outVTensor(d_topk_indices, {nq, k});
    
    // 调用 Faiss kernel
    faiss::gpu::runBlockSelect(
        inTensor, outKTensor, outVTensor,
        true,  // dir=true: 选择最大值（余弦相似度）
        k, stream
    );
}
```

### 2. 主程序调用

#### `main_test.cpp`
```cpp
// 步骤 6: GPU Top-K 选择
float *d_topk_distances;
int64_t *d_topk_indices;

cudaMalloc(&d_topk_distances, nq * k * sizeof(float));
cudaMalloc(&d_topk_indices, nq * k * sizeof(int64_t));

// 调用 GPU Top-K kernel
gpuTopKSelect(d_distances, d_topk_distances, d_topk_indices, 
              nq, nb, k, stream);

// 拷贝结果回 CPU
cudaMemcpy(h_topk_distances.data(), d_topk_distances, ...);
cudaMemcpy(h_topk_indices.data(), d_topk_indices, ...);
```

## Faiss blockSelect Kernel 详解

### Kernel 层次结构

```
runBlockSelect()  (Host 接口)
    ↓
blockSelect<>()  (__global__ kernel)
    ↓
BlockSelect<>  (模板类)
    ↓
WarpSelect<>  (Warp 级别选择)
```

### 算法原理

Faiss 的 `blockSelect` 使用**高效的 warp-level 选择算法**：

1. **分层队列**:
   - 每个线程维护一个小队列（ThreadQ）
   - 每个 warp 维护一个共享队列（WarpQ）

2. **增量合并**:
   - 线程处理数据流，动态添加到 ThreadQ
   - ThreadQ 满时，与 WarpQ 合并
   - 最终 reduce 产生 Top-K 结果

3. **内存优化**:
   - 使用寄存器存储队列（高速）
   - 使用 shared memory 做 warp 间通信
   - 避免全局内存原子操作

### 性能特点

| 特性 | 说明 |
|------|------|
| 复杂度 | O(n) 单次遍历 |
| 内存访问 | 合并访问，高效利用带宽 |
| 并行度 | Block-level 并行，每个 query 一个 block |
| 寄存器使用 | 根据 k 值自动选择最优配置 |
| 支持的 k 值 | 1, 32, 64, 128, 256, 512, 1024, 2048 |

### 配置选择

Faiss 根据 k 值自动选择最优配置：

```cpp
if (k == 1)           // 特殊优化：single-pass reduction
    BLOCK_SELECT_CALL(float, true, 1);
else if (k <= 32)     // Warp-level selection
    BLOCK_SELECT_CALL(float, true, 32);
else if (k <= 64)     // 2-warp merge
    BLOCK_SELECT_CALL(float, true, 64);
// ... 以此类推
```

## 完整的执行流程

```
┌─────────────────────────────────────────────────────┐
│ 1. 生成随机数据 (CPU)                                │
│    - Database: [nb × d]                             │
│    - Queries: [nq × d]                              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 2. 拷贝到 GPU (cudaMemcpy)                          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 3. GPU L2 归一化                                     │
│    Kernel: faiss::gpu::l2NormRowMajor               │
│    - 计算每个向量的 L2 范数                          │
│    - 除以范数使向量单位化                            │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 4. GPU 内积计算                                      │
│    Kernel: cublasSgemm                               │
│    - d_distances[nq × nb] = queries × database^T    │
│    - 利用 Tensor Cores 加速                          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 5. GPU Top-K 选择 ⭐                                 │
│    Kernel: faiss::gpu::blockSelect                   │
│    - 对每个 query 选择 top-k 个最大相似度           │
│    - 输出: distances[nq × k], indices[nq × k]      │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 6. 拷贝结果回 CPU (cudaMemcpy)                      │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 7. 验证和显示结果 (CPU)                             │
└─────────────────────────────────────────────────────┘
```

## 性能对比

### CPU vs GPU Top-K

| 方法 | 复杂度 | 数据传输 | 预期加速比 |
|------|--------|----------|------------|
| CPU `std::partial_sort` | O(n log k) × nq | GPU→CPU 全矩阵 | 1× (基线) |
| GPU `blockSelect` | O(n) × nq | GPU→CPU 仅 Top-K | 10-50× |

**示例**（nq=100, nb=10000, k=10）：
- CPU: 需传输 100×10000×4B = 3.81 MB 后排序
- GPU: 传输 100×10×4B = 3.9 KB，在 GPU 上排序
- **数据传输减少**: ~1000×
- **计算加速**: ~10-20×

## 关键优化技巧

### 1. 避免不必要的数据传输
```cpp
// ❌ 错误: 传输全矩阵到 CPU 再排序
cudaMemcpy(h_distances, d_distances, nq * nb * sizeof(float), ...);
cpu_sort(h_distances);

// ✅ 正确: GPU 上排序，只传输 Top-K
gpuTopKSelect(d_distances, d_topk, ...);
cudaMemcpy(h_topk, d_topk, nq * k * sizeof(float), ...);
```

### 2. 使用 Stream 重叠计算
```cpp
// 可以进一步优化：不同 query batch 用不同 stream
cudaStream_t streams[N];
for (int i = 0; i < N; i++) {
    gpuTopKSelect(..., streams[i]);
}
```

### 3. 选择合适的 k 值
- k ≤ 32: 最优（单 warp）
- k ≤ 1024: 良好
- k > 1024: 考虑其他方法（cuVS）

## 编译和运行

确保链接到 Faiss 的 `runBlockSelect`:

```bash
cd cosine_similarity_test/build
cmake ..
make -j$(nproc)
./main_test
```

预期输出：
```
[步骤 6] GPU 端 Top-K 选择...
  直接调用 Faiss 的 blockSelect __global__ kernel
  ✓ GPU Top-K 选择完成 (耗时: X.XX ms)
```

## 相关文件

- `faiss/gpu/utils/BlockSelectKernel.cuh` - blockSelect kernel 定义
- `faiss/gpu/utils/BlockSelectFloat.cu` - float 类型实例化
- `faiss/gpu/utils/Select.cuh` - WarpSelect 和 BlockSelect 实现
- `cosine_similarity_test/gpu_normalize.cu` - 我们的封装函数
- `cosine_similarity_test/main_test.cpp` - 主程序

## 进一步优化方向

1. **使用 cuVS backend** (k > 2048)
2. **Batch processing** (处理超大 query 集)
3. **Multi-GPU** (分布式搜索)
4. **混合精度** (FP16/INT8 量化)

## 总结

现在我们的实现**完全使用 Faiss 的底层 GPU kernel**，包括：
- ✅ L2 归一化: `l2NormRowMajor`
- ✅ 内积计算: `cublasSgemm`
- ✅ Top-K 选择: `blockSelect` ⭐

这是 Faiss GPU 索引内部使用的**相同 kernel**，可以测试 Faiss 的真实极限性能！🚀

