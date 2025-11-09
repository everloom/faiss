# GEMM + Top-K Fusion 实现方案

## 📊 Fusion 可行性分析

### 当前的两阶段流程

```
阶段 1: GEMM (cuBLAS)
  输入: queries[nq × d], database[nb × d]
  输出: distances[nq × nb] 完整矩阵
  
  内存写入: nq × nb × 4 bytes (例如 8 × 60000 × 4 = 1.92 MB)
  ↓
阶段 2: Top-K (Faiss blockSelect)
  输入: distances[nq × nb] 完整矩阵
  输出: topk_values[nq × k], topk_indices[nq × k]
  
  内存读取: nq × nb × 4 bytes (1.92 MB)
  内存写入: nq × k × 8 bytes (例如 8 × 2048 × 8 = 128 KB)
```

**性能瓶颈**：
- 中间矩阵占用大量内存带宽
- 对于 nq=8, nb=60000, k=2048：
  - 写入 1.92 MB 后立即读取 1.92 MB
  - 但实际只需要 128 KB 的结果
  - **带宽浪费：15×**

### Fusion 后的单阶段流程

```
Fused GEMM+TopK Kernel
  输入: queries[nq × d], database[nb × d]
  输出: topk_values[nq × k], topk_indices[nq × k]
  
  内存写入: nq × k × 8 bytes (128 KB)
  
  中间结果: 在寄存器/shared memory 中，不写 global memory
```

**性能提升预估**：
- 内存带宽减少：**~15-30×**
- 总延迟降低：**10-25%** (对于 memory-bound 的场景)

## 🔧 三种实现方案

### 方案 1: 简单融合版（已实现）

**特点**：
- 每个 block 处理一个 query
- 每个线程遍历所有 database 向量
- 在寄存器中维护 Top-K

**优点**：
- 实现简单
- 内存占用小

**缺点**：
- GEMM 计算未优化（无 Tiling，无 Tensor Core）
- 适合小规模数据

**适用场景**：
- d ≤ 256
- nb ≤ 100K
- k ≤ 128

### 方案 2: Tensor Core 优化版（框架已给出）

**特点**：
- 使用 WMMA API（Tensor Core）
- FP16 输入，FP32 累加
- 边计算边更新 Top-K

**优点**：
- 利用 Tensor Core，GEMM 性能提升 **8-16×**
- 仍然避免写中间矩阵

**缺点**：
- 需要 FP16 输入数据
- Tensor Core 的 fragment 布局复杂

**适用场景**：
- Volta/Turing/Ampere/Hopper GPU
- 对精度要求不极端高的场景

### 方案 3: 基于你的 MMA Kernel 改造（**推荐**）⭐

查看你的 `hgemm_mma_stage.cu`，我建议这样改造：

#### 原始 GEMM Kernel 结构
```cpp
hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel() {
    // 1. 加载 A, B tiles 到 shared memory (staging)
    // 2. 使用 WMMA 计算 C fragment
    // 3. 将 C fragment 写回 global memory  ← 修改这里！
}
```

#### 改造方案：在 Epilogue 融入 Top-K

```cuda
template<int M, int N, int K, int TOP_K>
__global__ void hgemmTopKFused(
    const half* A,     // queries [nq, d]
    const half* B,     // database [nb, d]  
    float* topk_vals,  // [nq, TOP_K]
    int* topk_idxs,    // [nq, TOP_K]
    int nq, int nb, int d
) {
    // ===== GEMM 部分 (保持不变) =====
    
    // Tile indices
    int bx = blockIdx.x;
    int by = blockIdx.y;  // 这里 by 对应 query index
    
    // Shared memory for staging
    __shared__ half smem_A[...];
    __shared__ half smem_B[...];
    
    // WMMA fragments
    wmma::fragment<...> frag_A, frag_B, frag_C;
    wmma::fill_fragment(frag_C, 0.0f);
    
    // Multi-stage pipeline
    for (int stage = 0; stage < num_stages; stage++) {
        // Load tiles
        // MMA computation
        wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
    }
    
    // ===== 修改：Epilogue 部分 =====
    // 原来：写 C fragment 到 global memory
    // 现在：将 C fragment 结果添加到 Top-K 队列
    
    __shared__ float smem_topk_vals[WARPS_PER_BLOCK * TOP_K];
    __shared__ int smem_topk_idxs[WARPS_PER_BLOCK * TOP_K];
    
    // 每个线程维护局部 Top-K
    WarpTopK<TOP_K> thread_topk;
    
    // 从 frag_C 中提取结果
    for (int i = 0; i < frag_C.num_elements; i++) {
        int col_idx = bx * N + get_col_from_fragment(i);
        if (col_idx < nb) {
            thread_topk.add(frag_C.x[i], col_idx);
        }
    }
    
    // Warp reduce
    thread_topk.warpReduce();
    
    // Block reduce
    // ... (同方案1)
    
    // 写出 Top-K
    if (threadIdx.x == 0) {
        for (int i = 0; i < TOP_K; i++) {
            topk_vals[by * TOP_K + i] = final_topk.values[i];
            topk_idxs[by * TOP_K + i] = final_topk.indices[i];
        }
    }
}
```

## 🎯 关键优化技术

### 1. 在线 Top-K 更新

```cpp
// 不要：计算所有结果 → 排序
for (int i = 0; i < n; i++) {
    results[i] = compute(i);
}
sort(results);  // O(n log n)

// 要：增量更新 Top-K
TopK queue;
for (int i = 0; i < n; i++) {
    float val = compute(i);
    queue.add(val, i);  // O(k)，只保留 Top-K
}
// 总复杂度：O(n × k)，但 k << n
```

### 2. 分层归约

```
Thread Level:  每个线程维护 Top-K (寄存器)
     ↓
Warp Level:    Warp 内归约 (shuffle)
     ↓
Block Level:   Block 内归约 (shared memory)
     ↓
Output:        写出最终 Top-K
```

### 3. 内存访问优化

| 版本 | Global Memory 写入 | Global Memory 读取 |
|------|-------------------|-------------------|
| 原始（分离） | nq×nb + nq×k×2 | nq×nb |
| Fusion | nq×k×2 | 0 (中间结果) |
| **节省** | **~15-30×** | **完全消除** |

## 📈 性能对比预测

假设：nq=8, nb=60000, d=256, k=2048

### 原始方案
```
GEMM:   1.92 MB write + cuBLAS 计算      ~0.15 ms
TopK:   1.92 MB read + blockSelect       ~1.5 ms
------------------------------------------------------
Total:  3.84 MB 传输 + 计算               ~1.65 ms
```

### Fusion 方案
```
Fused:  128 KB write + 融合计算          ~0.12 ms
------------------------------------------------------
Total:  128 KB 传输 + 计算                ~0.12 ms

加速比: 1.65 / 0.12 ≈ 13.75×
```

**实际加速比**：取决于计算/内存比
- Memory-bound (小维度): **10-20×**
- Compute-bound (大维度): **1.2-2×**

## 🚀 实现建议

### Step 1: 简单版本验证（使用 gemm_topk_fused.cu）

1. 先实现简单版本验证正确性
2. 对比性能：融合 vs 分离

### Step 2: 优化版本（基于你的 MMA kernel）

1. 修改 `hgemm_mma_stage.cu` 的 epilogue
2. 替换写 global memory → 更新 Top-K queue
3. 添加 block-wide Top-K 归约

### Step 3: 进一步优化

1. **使用 Tensor Core**: FP16 GEMM + FP32 Top-K
2. **Warp 特化**: 不同 k 值使用不同的 warp 配置
3. **Double buffering**: Top-K 队列使用双缓冲
4. **自适应算法**: 根据 nb/k 比例选择 fusion 或分离

## 💡 何时使用 Fusion？

**建议使用 Fusion**：
- ✅ k 相对较小 (k ≤ 2048)
- ✅ nb 很大 (nb ≥ 10K)
- ✅ 内存带宽受限

**建议使用分离**：
- ❌ k 很大 (k > 2048，超过 Faiss 限制)
- ❌ 需要保存完整距离矩阵（后续分析用）
- ❌ nq 很大，nb 很小（此时 GEMM 不是瓶颈）

## 🔨 集成到 main_test.cpp

```cpp
// 添加 fusion 选项
bool use_fusion = true;  // 切换 fusion/分离模式

if (use_fusion) {
    // 直接调用 fused kernel
    launchGemmTopKFused(
        d_queries, d_database,
        d_topk_distances, d_topk_indices,
        config.query_bs, config.db_bs, config.seq_len,
        config.top_k, stream
    );
} else {
    // 原来的两步法
    cublasSgemm(...);
    gpuTopKSelect(...);
}
```

## 📚 参考资料

- **Faiss GPU 论文**: [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
- **Flash Attention**: 类似的 fusion 思想（Softmax + Attention）
- **Online algorithms**: 流式 Top-K 算法

## 总结

✅ **GEMM + Top-K Fusion 完全可行**
✅ **性能提升显著**（特别是 memory-bound 场景）
✅ **实现复杂度中等**（需要仔细处理 Top-K 归约）

关键是在 GEMM 的 **epilogue 阶段**，不写 global memory，而是直接更新 Top-K 队列！

