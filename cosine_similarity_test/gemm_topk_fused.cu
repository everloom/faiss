/*
 * GEMM + Top-K Fusion Kernel
 * 将 GEMM 计算和 Top-K 选择融合在一个 kernel 中
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>

using namespace nvcuda;

// ============================================================================
// 简化版 WarpSelect：在寄存器中维护 Top-K
// ============================================================================

template<int K>
struct WarpTopK {
    float values[K];
    int indices[K];
    
    __device__ inline WarpTopK() {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            values[i] = -INFINITY;
            indices[i] = -1;
        }
    }
    
    // 添加一个新元素到 Top-K
    __device__ inline void add(float val, int idx) {
        // 找到插入位置（降序排列，最大的在前面）
        if (val <= values[K-1]) return;  // 比最小的还小，直接丢弃
        
        int pos = K - 1;
        #pragma unroll
        for (int i = 0; i < K - 1; i++) {
            if (val > values[i]) {
                pos = i;
                break;
            }
        }
        
        // 插入：右移后面的元素
        #pragma unroll
        for (int i = K - 1; i > pos; i--) {
            values[i] = values[i - 1];
            indices[i] = indices[i - 1];
        }
        
        values[pos] = val;
        indices[pos] = idx;
    }
    
    // Warp 内归约：合并多个线程的 Top-K
    __device__ inline void warpReduce() {
        // 每个线程有 K 个元素，warp 有 32 个线程
        // 需要从 32×K 个元素中选出 K 个最大的
        
        // 简化实现：使用 shuffle 进行成对合并
        for (int offset = 16; offset > 0; offset /= 2) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                float other_val = __shfl_down_sync(0xffffffff, values[i], offset);
                int other_idx = __shfl_down_sync(0xffffffff, indices[i], offset);
                
                if (other_val > values[K-1]) {
                    add(other_val, other_idx);
                }
            }
        }
    }
};

// ============================================================================
// GEMM + Top-K Fused Kernel
// ============================================================================

/*
 * 每个 block 负责计算一个 query 的所有结果，并直接选出 Top-K
 * 
 * 参数：
 *   - queries: [nq, d] 查询向量
 *   - database: [nb, d] 数据库向量
 *   - topk_values: [nq, k] 输出 Top-K 相似度
 *   - topk_indices: [nq, k] 输出 Top-K 索引
 */
template<int TILE_SIZE, int TOP_K>
__global__ void gemmTopKFused(
    const float* __restrict__ queries,   // [nq, d]
    const float* __restrict__ database,  // [nb, d]
    float* __restrict__ topk_values,     // [nq, k]
    int* __restrict__ topk_indices,      // [nq, k]
    int nq,    // query 数量
    int nb,    // database 数量
    int d      // 向量维度
) {
    // 每个 block 处理一个 query
    int query_idx = blockIdx.x;
    if (query_idx >= nq) return;
    
    // 每个线程维护一个 Top-K 队列
    WarpTopK<TOP_K> thread_topk;
    
    // 共享内存：存储 query 向量和 database tile
    __shared__ float smem_query[TILE_SIZE];
    __shared__ float smem_db[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict
    
    // 加载 query 向量到共享内存
    const float* query_ptr = queries + query_idx * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        smem_query[i] = query_ptr[i];
    }
    __syncthreads();
    
    // 遍历所有 database 向量
    for (int db_start = 0; db_start < nb; db_start += TILE_SIZE) {
        int db_end = min(db_start + TILE_SIZE, nb);
        
        // 每个线程计算一个或多个 database 向量的内积
        for (int db_idx = db_start + threadIdx.x; db_idx < db_end; db_idx += blockDim.x) {
            const float* db_ptr = database + db_idx * d;
            
            // 计算内积（余弦相似度）
            float dot_product = 0.0f;
            
            #pragma unroll 4
            for (int i = 0; i < d; i++) {
                dot_product += smem_query[i] * db_ptr[i];
            }
            
            // 立即添加到 Top-K 队列（在寄存器中）
            thread_topk.add(dot_product, db_idx);
        }
    }
    
    // Warp 内归约：合并 32 个线程的 Top-K
    thread_topk.warpReduce();
    
    // 使用 shared memory 进行 block 级归约
    __shared__ float smem_topk_vals[32 * TOP_K];  // 最多 32 个 warp
    __shared__ int smem_topk_idxs[32 * TOP_K];
    
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Lane 0 写入 shared memory
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            smem_topk_vals[warp_id * TOP_K + i] = thread_topk.values[i];
            smem_topk_idxs[warp_id * TOP_K + i] = thread_topk.indices[i];
        }
    }
    __syncthreads();
    
    // 使用第一个 warp 做最终归约
    if (threadIdx.x < 32) {
        WarpTopK<TOP_K> final_topk;
        
        // 从所有 warp 的结果中选择
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            #pragma unroll
            for (int i = 0; i < TOP_K; i++) {
                float val = smem_topk_vals[w * TOP_K + i];
                int idx = smem_topk_idxs[w * TOP_K + i];
                if (idx >= 0) {  // 有效元素
                    final_topk.add(val, idx);
                }
            }
        }
        
        // Lane 0 写出最终结果
        if (lane_id == 0) {
            float* out_vals = topk_values + query_idx * TOP_K;
            int* out_idxs = topk_indices + query_idx * TOP_K;
            
            #pragma unroll
            for (int i = 0; i < TOP_K; i++) {
                out_vals[i] = final_topk.values[i];
                out_idxs[i] = final_topk.indices[i];
            }
        }
    }
}

// ============================================================================
// 使用 Tensor Core 的优化版本（FP16 GEMM + FP32 Top-K）
// ============================================================================

template<int TOP_K, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void gemmTopKFusedTensorCore(
    const half* __restrict__ queries,    // [nq, d] FP16
    const half* __restrict__ database,   // [nb, d] FP16
    float* __restrict__ topk_values,     // [nq, k] FP32
    int* __restrict__ topk_indices,      // [nq, k]
    int nq, int nb, int d
) {
    // 每个 block 处理一个 query
    int query_idx = blockIdx.x;
    if (query_idx >= nq) return;
    
    // 使用 WMMA API
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    WarpTopK<TOP_K> thread_topk;
    
    // 每个 warp 处理一部分 database 向量
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    
    for (int db_tile = warp_id; db_tile < nb; db_tile += num_warps * WMMA_N) {
        // 使用 Tensor Core 计算
        wmma::fill_fragment(c_frag, 0.0f);
        
        for (int k_tile = 0; k_tile < d; k_tile += WMMA_K) {
            // 加载 query 和 database tiles
            wmma::load_matrix_sync(a_frag, queries + query_idx * d + k_tile, d);
            wmma::load_matrix_sync(b_frag, database + db_tile * d + k_tile, d);
            
            // Tensor Core MMA
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        // 从 accumulator fragment 中提取结果并添加到 Top-K
        // 注意：这里简化了，实际需要处理 fragment 的布局
        for (int i = 0; i < c_frag.num_elements; i++) {
            int db_idx = db_tile + i;
            if (db_idx < nb) {
                thread_topk.add(c_frag.x[i], db_idx);
            }
        }
    }
    
    // 后续 Top-K 归约逻辑同上...
}

// ============================================================================
// 更实用的实现：分块 GEMM + 在线 Top-K
// ============================================================================

/*
 * 核心思想：
 * 1. 将 database 分成多个 tile
 * 2. 对每个 tile 计算 GEMM，得到部分结果
 * 3. 立即对部分结果进行 Top-K 更新
 * 4. 丢弃不在 Top-K 中的结果，减少内存占用
 */
template<int BLOCK_SIZE, int TOP_K>
__global__ void gemmTopKTiled(
    const float* __restrict__ queries,   // [nq, d]
    const float* __restrict__ database,  // [nb, d]
    float* __restrict__ topk_values,     // [nq, k]
    int* __restrict__ topk_indices,      // [nq, k]
    int nq, int nb, int d
) {
    // 每个 block 处理一个 query
    int query_idx = blockIdx.x;
    if (query_idx >= nq) return;
    
    // 每个线程维护 Top-K
    WarpTopK<TOP_K> topk;
    
    // 加载 query 到寄存器/shared memory
    extern __shared__ float smem[];
    float* smem_query = smem;
    
    const float* query = queries + query_idx * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        smem_query[i] = query[i];
    }
    __syncthreads();
    
    // 每个线程处理一部分 database 向量
    for (int db_idx = threadIdx.x; db_idx < nb; db_idx += blockDim.x) {
        const float* db_vec = database + db_idx * d;
        
        // 计算内积
        float dot = 0.0f;
        
        // 向量化加载和计算
        #pragma unroll 4
        for (int i = 0; i < d; i++) {
            dot += smem_query[i] * db_vec[i];
        }
        
        // 立即更新 Top-K（在寄存器中）
        topk.add(dot, db_idx);
    }
    
    // Warp 内归约
    topk.warpReduce();
    
    // Block 级归约（使用 shared memory）
    __shared__ float smem_vals[32 * TOP_K];
    __shared__ int smem_idxs[32 * TOP_K];
    
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            smem_vals[warp_id * TOP_K + i] = topk.values[i];
            smem_idxs[warp_id * TOP_K + i] = topk.indices[i];
        }
    }
    __syncthreads();
    
    // 第一个 warp 做最终合并
    if (warp_id == 0) {
        WarpTopK<TOP_K> final_topk;
        int num_warps = (blockDim.x + 31) / 32;
        
        // 从所有 warp 收集结果
        for (int w = lane_id; w < num_warps * TOP_K; w += 32) {
            if (smem_idxs[w] >= 0) {
                final_topk.add(smem_vals[w], smem_idxs[w]);
            }
        }
        
        // Warp 内最终归约
        final_topk.warpReduce();
        
        // Lane 0 写出结果
        if (lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < TOP_K; i++) {
                topk_values[query_idx * TOP_K + i] = final_topk.values[i];
                topk_indices[query_idx * TOP_K + i] = final_topk.indices[i];
            }
        }
    }
}

// ============================================================================
// 基于已有 GEMM kernel 的改进方案
// ============================================================================

/*
 * 方案：修改 GEMM kernel，在计算出每行结果后立即进行 Top-K
 * 
 * 关键修改点：
 * 1. 输出不再写完整矩阵，而是维护 Top-K 队列
 * 2. 在 GEMM 的 epilogue 阶段融入 Top-K 逻辑
 * 3. 每个 warp 计算一部分 database，并更新局部 Top-K
 * 4. 最后进行 block-wide 归约
 */

// Host 接口
extern "C" void launchGemmTopKFused(
    const float* d_queries,
    const float* d_database,
    float* d_topk_values,
    int* d_topk_indices,
    int nq, int nb, int d,
    int k,
    cudaStream_t stream
) {
    // 根据 k 值选择合适的模板实例
    constexpr int BLOCK_SIZE = 256;
    
    dim3 grid(nq);
    dim3 block(BLOCK_SIZE);
    
    // 共享内存大小：存储 query 向量
    int smem_size = d * sizeof(float);
    
    if (k <= 10) {
        gemmTopKTiled<BLOCK_SIZE, 10><<<grid, block, smem_size, stream>>>(
            d_queries, d_database, d_topk_values, d_topk_indices, nq, nb, d
        );
    } else if (k <= 32) {
        gemmTopKTiled<BLOCK_SIZE, 32><<<grid, block, smem_size, stream>>>(
            d_queries, d_database, d_topk_values, d_topk_indices, nq, nb, d
        );
    } else if (k <= 64) {
        gemmTopKTiled<BLOCK_SIZE, 64><<<grid, block, smem_size, stream>>>(
            d_queries, d_database, d_topk_values, d_topk_indices, nq, nb, d
        );
    } else if (k <= 128) {
        gemmTopKTiled<BLOCK_SIZE, 128><<<grid, block, smem_size, stream>>>(
            d_queries, d_database, d_topk_values, d_topk_indices, nq, nb, d
        );
    }
    // ... 其他 k 值
}

// ============================================================================
// 性能优化版本：使用更高效的 GEMM 实现
// ============================================================================

/*
 * 利用你提供的 hgemm_mma_stage kernel 的思想：
 * 
 * 1. 使用 Tensor Core 加速 GEMM 计算
 * 2. 使用 double buffering/staging 隐藏内存延迟
 * 3. 在每个 stage 计算完成后，立即更新 Top-K
 * 4. 使用高效的归并网络进行 Top-K 归约
 * 
 * 伪代码：
 */
/*
__global__ void gemmTopKFusedOptimized(...) {
    // Stage 1: 加载第一个 tile
    loadTile(stage0);
    
    for (int stage = 0; stage < num_stages; stage++) {
        // Async 加载下一个 stage（隐藏延迟）
        if (stage + 1 < num_stages) {
            loadTileAsync(stage + 1);
        }
        
        // 使用 Tensor Core 计算当前 stage
        computeGemmTile(stage);
        
        // 立即将结果更新到 Top-K（不写 global memory）
        updateTopK(gemm_result, topk_queue);
    }
    
    // 最终归约和输出
    reduceTopK();
    writeOutput();
}
*/

