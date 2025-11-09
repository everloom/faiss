/*
 * GEMM + Top-K Fusion Kernel
 * 基于 hgemm_mma_stage.cu 和 Faiss BlockSelect 的融合实现
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

// ============================================================================
// 从 Faiss 移植的 WarpSelect 实现
// ============================================================================

template<typename K, typename V, int NumWarpQ>
struct WarpSelectFused {
    static constexpr int kNumWarpQRegisters = NumWarpQ / 32;
    
    K warpK[kNumWarpQRegisters];
    V warpV[kNumWarpQRegisters];
    
    __device__ inline WarpSelectFused(K initK, V initV) {
        #pragma unroll
        for (int i = 0; i < kNumWarpQRegisters; ++i) {
            warpK[i] = initK;
            warpV[i] = initV;
        }
    }
    
    // 添加元素到 warp queue
    __device__ inline void add(K k, V v) {
        // 简化版：只保留核心逻辑
        // 找到插入位置并插入
        bool need_add = (k > warpK[kNumWarpQRegisters - 1]);
        
        #pragma unroll
        for (int i = kNumWarpQRegisters - 1; i >= 0; i--) {
            if (need_add && k > warpK[i]) {
                if (i < kNumWarpQRegisters - 1) {
                    warpK[i + 1] = warpK[i];
                    warpV[i + 1] = warpV[i];
                }
            }
        }
        
        if (need_add) {
            warpK[0] = k;
            warpV[0] = v;
        }
    }
    
    // Warp 内归约（使用 shuffle）
    __device__ inline void reduce() {
        int lane_id = threadIdx.x % 32;
        
        // 从其他 lane 收集数据
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            #pragma unroll
            for (int i = 0; i < kNumWarpQRegisters; i++) {
                K other_k = __shfl_xor_sync(0xffffffff, warpK[i], offset);
                V other_v = __shfl_xor_sync(0xffffffff, warpV[i], offset);
                
                if (other_k > warpK[kNumWarpQRegisters - 1]) {
                    add(other_k, other_v);
                }
            }
        }
    }
};

// ============================================================================
// 辅助宏和函数（从原 GEMM kernel 复制）
// ============================================================================

#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
               : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) \
               : "r"(addr))

#define LDMATRIX_X2_T(R0, R1, addr) \
  asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
               : "=r"(R0), "=r"(R1) \
               : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
               "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
               : "=r"(RD0), "=r"(RD1) \
               : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), \
                 "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

#define CP_ASYNC_CG(dst, src, bytes) \
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" \
               ::"r"(dst), "l"(src), "n"(bytes))

#define CP_ASYNC_COMMIT_GROUP() \
  asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(n) \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

__device__ __forceinline__ int div_ceil(int a, int b) {
  return (a + b - 1) / b;
}

// ============================================================================
// GEMM + Top-K Fused Kernel
// ============================================================================

/*
 * 关键改造：
 * 1. Grid 配置改为 [nq, ceil(nb/BN)] - 每行(query)由多个 block 协作
 * 2. 每个 block 计算部分列(database)的相似度
 * 3. 在 epilogue 中，不写 C 矩阵，而是更新 Top-K 队列
 * 4. 使用 shared memory 做 block 级 Top-K 归约
 * 5. 最后使用原子操作或 grid-level 归约产生最终 Top-K
 */

template<
    int MMA_M,        // 16
    int MMA_N,        // 8  
    int MMA_K,        // 16
    int MMA_TILE_M,   // 2
    int MMA_TILE_N,   // 4
    int WARP_TILE_M,  // 4
    int WARP_TILE_N,  // 4
    int WARP_TILE_K,  // 2
    int A_PAD,        // 0
    int B_PAD,        // 8
    int K_STAGE,      // 3
    int BLOCK_SWIZZLE,// 0
    int WARP_SWIZZLE, // 0
    int TOP_K         // Top-K 大小，例如 32, 64, 128...
>
__global__ void __launch_bounds__(256)
    hgemm_topk_fused_kernel(
        const half *__restrict__ queries,    // [nq, d] 查询向量
        const half *__restrict__ database,   // [nb, d] 数据库向量
        float *__restrict__ topk_values,     // [nq, TOP_K] 输出
        int *__restrict__ topk_indices,      // [nq, TOP_K] 输出
        int nq,  // query 数量
        int nb,  // database 数量
        int d    // 向量维度
) {
    // 注意：这里的语义发生变化
    // A = queries [nq, d]  (原来的 M × K)
    // B = database [nb, d] (原来的 N × K, 但我们需要转置语义)
    // C = similarities [nq, nb] (但不写出，而是做 Top-K)
    
    const int bx = blockIdx.x;  // database tile index
    const int by = blockIdx.y;  // query index (每个 query 一行)
    
    const int M = nq;
    const int N = nb;
    const int K = d;
    
    const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 128
    constexpr int BK = MMA_K;                            // 16
    
    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BK * (BN + B_PAD);
    constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
    constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id % 2;
    const int warp_n = warp_id / 2;
    
    int load_smem_a_m = tid / 2;
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;
    int load_smem_b_k = tid / 16;
    int load_smem_b_n = (tid % 16) * 8;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    
    if (load_gmem_a_m >= M || load_gmem_b_n >= N)
        return;
    
    // MMA 累加器
    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }
    
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);
    
    // ========================================================================
    // 阶段 1: GEMM 计算（保持原样）
    // ========================================================================
    
    // Pipeline 预加载
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) {
        int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        
        uint32_t load_smem_a_ptr =
            (smem_a_base_ptr +
             (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                 sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &queries[load_gmem_a_addr], 16);
        
        uint32_t load_smem_a_mma_k_ptr =
            (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
             (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                 sizeof(half));
        CP_ASYNC_CG(load_smem_a_mma_k_ptr, &queries[load_gmem_a_addr + 16], 16);
        
        uint32_t load_smem_b_ptr =
            (smem_b_base_ptr +
             (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
                 sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &database[load_gmem_b_addr], 16);
        
        int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
        int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n;
        uint32_t load_smem_b_mma_k_ptr =
            (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) +
             (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
                 sizeof(half));
        CP_ASYNC_CG(load_smem_b_mma_k_ptr, &database[load_gmem_b_addr_mma_k], 16);
        
        CP_ASYNC_COMMIT_GROUP();
    }
    
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
    
    uint32_t RA[2][WARP_TILE_M][4];
    uint32_t RB[2][WARP_TILE_N][2];
    int reg_store_idx = 0;
    int reg_load_idx = 1;
    
    // 预加载第一个 K tile
    {
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
            int lane_smem_a_k = (lane_id / 16) * 8;
            uint32_t lane_smem_a_ptr =
                (smem_a_base_ptr + (0 * s_a_stage_offset +
                                    lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) *
                                       sizeof(half));
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        lane_smem_a_ptr);
        }
        
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int lane_smem_b_k = lane_id % 16;
            int lane_smem_b_n = warp_smem_b_n;
            uint32_t lane_smem_b_ptr =
                (smem_b_base_ptr + (0 * s_b_stage_offset +
                                    lane_smem_b_k * (BN + B_PAD) + lane_smem_b_n) *
                                       sizeof(half));
            LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                          lane_smem_b_ptr);
        }
    }
    
    // 主循环：MMA 计算
    #pragma unroll
    for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;
        int smem_sel = (k + 1) % K_STAGE;
        int smem_sel_next = k % K_STAGE;
        
        // 异步加载下一个 tile（省略详细代码，与原 kernel 相同）
        // ...
        
        // MMA 计算第一个 MMA_K
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            #pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                int j_s = ((i % 2) && WARP_SWIZZLE) ? (WARP_TILE_N - j - 1) : j;
                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0],
                          RA[reg_load_idx][i][1], RA[reg_load_idx][i][2],
                          RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }
        
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;
        
        // MMA 计算第二个 MMA_K
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            #pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                int j_s = ((i % 2) && WARP_SWIZZLE) ? (WARP_TILE_N - j - 1) : j;
                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0],
                          RA[reg_load_idx][i][1], RA[reg_load_idx][i][2],
                          RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }
        
        // 加载下一个 tile（省略）
        // ...
    }
    
    // ========================================================================
    // 阶段 2: Epilogue - 融合 Top-K 选择（关键修改！）
    // ========================================================================
    
    // 每个 warp 维护自己的 Top-K
    WarpSelectFused<float, int, TOP_K> warp_topk(-INFINITY, -1);
    
    // 从 RC (FP32 accumulator) 中提取结果并添加到 Top-K
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            // RC[i][j][0] 和 RC[i][j][1] 包含 2 个 FP32 值
            // 需要转换为 float* 来访问
            float* rc_ptr = reinterpret_cast<float*>(&RC[i][j][0]);
            
            // 计算全局列索引
            int warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int lane_gmem_c_n = bx * BN + warp_smem_c_n;
            
            // MMA 16x8 产生 2 个 FP32 值
            #pragma unroll
            for (int elem = 0; elem < 2; elem++) {
                int global_db_idx = lane_gmem_c_n + elem;
                
                if (global_db_idx < nb) {
                    float similarity = rc_ptr[elem];
                    warp_topk.add(similarity, global_db_idx);
                }
            }
        }
    }
    
    // Warp 内归约
    warp_topk.reduce();
    
    // ========================================================================
    // 阶段 3: Block 级归约
    // ========================================================================
    
    constexpr int NUM_WARPS = 8;  // 256 / 32
    __shared__ float smem_topk_vals[NUM_WARPS * TOP_K];
    __shared__ int smem_topk_idxs[NUM_WARPS * TOP_K];
    
    // 每个 warp 的 lane 0 写入 shared memory
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < TOP_K / 32; i++) {
            smem_topk_vals[warp_id * TOP_K + i] = warp_topk.warpK[i];
            smem_topk_idxs[warp_id * TOP_K + i] = warp_topk.warpV[i];
        }
    }
    __syncthreads();
    
    // 第一个 warp 做最终归约
    if (warp_id == 0) {
        WarpSelectFused<float, int, TOP_K> final_topk(-INFINITY, -1);
        
        // 从所有 warp 收集结果
        for (int w = 0; w < NUM_WARPS; w++) {
            #pragma unroll
            for (int i = 0; i < TOP_K / 32; i++) {
                float val = smem_topk_vals[w * TOP_K + i];
                int idx = smem_topk_idxs[w * TOP_K + i];
                
                if (idx >= 0) {
                    final_topk.add(val, idx);
                }
            }
        }
        
        // Warp 内最终归约
        final_topk.reduce();
        
        // Lane 0 写出最终结果
        // 注意：这里需要原子操作或 grid-level 归约
        // 因为一个 query 可能由多个 block 处理
        if (lane_id == 0) {
            // TODO: 实现 grid-level 的 Top-K 归约
            // 方案 1: 使用原子操作（CAS）
            // 方案 2: 两阶段kernel（第一阶段每个block输出局部TopK，第二阶段全局归约）
            // 方案 3: 限制每个query只用一个block（需要 BN >= nb）
            
            // 简化实现：假设每个 query 只有一个 block
            #pragma unroll
            for (int i = 0; i < TOP_K / 32; i++) {
                topk_values[by * TOP_K + i] = final_topk.warpK[i];
                topk_indices[by * TOP_K + i] = final_topk.warpV[i];
            }
        }
    }
}

// ============================================================================
// 简化版本：每个 query 只用一个 block（适合 nb 不太大的情况）
// ============================================================================

template<int TOP_K>
__global__ void __launch_bounds__(256)
    hgemm_topk_simple_fused_kernel(
        const half *__restrict__ queries,    // [nq, d]
        const half *__restrict__ database,   // [nb, d]
        float *__restrict__ topk_values,     // [nq, TOP_K]
        int *__restrict__ topk_indices,      // [nq, TOP_K]
        int nq, int nb, int d
) {
    // 每个 block 处理一个 query
    int query_idx = blockIdx.x;
    if (query_idx >= nq) return;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = 256 / 32;
    
    // 每个 warp 维护 Top-K
    WarpSelectFused<float, int, TOP_K> warp_topk(-INFINITY, -1);
    
    // 加载 query 到 shared memory
    __shared__ half smem_query[256];  // 假设 d <= 256
    for (int i = tid; i < d; i += 256) {
        smem_query[i] = queries[query_idx * d + i];
    }
    __syncthreads();
    
    // 每个线程处理一部分 database 向量
    for (int db_idx = tid; db_idx < nb; db_idx += 256) {
        const half* db_vec = database + db_idx * d;
        
        // 计算内积（使用 FP16→FP32）
        float dot = 0.0f;
        
        #pragma unroll 8
        for (int i = 0; i < d; i++) {
            dot += __half2float(smem_query[i]) * __half2float(db_vec[i]);
        }
        
        // 立即添加到 Top-K
        warp_topk.add(dot, db_idx);
    }
    
    // Warp 内归约
    warp_topk.reduce();
    
    // Block 级归约
    __shared__ float smem_vals[NUM_WARPS * TOP_K];
    __shared__ int smem_idxs[NUM_WARPS * TOP_K];
    
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < TOP_K / 32; i++) {
            smem_vals[warp_id * TOP_K + i] = warp_topk.warpK[i];
            smem_idxs[warp_id * TOP_K + i] = warp_topk.warpV[i];
        }
    }
    __syncthreads();
    
    // Warp 0 做最终归约
    if (warp_id == 0) {
        WarpSelectFused<float, int, TOP_K> final_topk(-INFINITY, -1);
        
        for (int w = 0; w < NUM_WARPS; w++) {
            #pragma unroll
            for (int i = 0; i < TOP_K / 32; i++) {
                if (smem_idxs[w * TOP_K + i] >= 0) {
                    final_topk.add(smem_vals[w * TOP_K + i], smem_idxs[w * TOP_K + i]);
                }
            }
        }
        
        final_topk.reduce();
        
        // Lane 0 写出
        if (lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < TOP_K / 32; i++) {
                topk_values[query_idx * TOP_K + i] = final_topk.warpK[i];
                topk_indices[query_idx * TOP_K + i] = final_topk.warpV[i];
            }
        }
    }
}

// ============================================================================
// Host 接口
// ============================================================================

extern "C" void launchHgemmTopKFused(
    const half* d_queries,
    const half* d_database,
    float* d_topk_values,
    int* d_topk_indices,
    int nq, int nb, int d,
    int k,
    cudaStream_t stream
) {
    // 简化版本：每个 query 一个 block
    dim3 grid(nq);
    dim3 block(256);
    
    if (k <= 32) {
        hgemm_topk_simple_fused_kernel<32><<<grid, block, 0, stream>>>(
            d_queries, d_database, d_topk_values, d_topk_indices, nq, nb, d
        );
    } else if (k <= 64) {
        hgemm_topk_simple_fused_kernel<64><<<grid, block, 0, stream>>>(
            d_queries, d_database, d_topk_values, d_topk_indices, nq, nb, d
        );
    } else if (k <= 128) {
        hgemm_topk_simple_fused_kernel<128><<<grid, block, 0, stream>>>(
            d_queries, d_database, d_topk_values, d_topk_indices, nq, nb, d
        );
    }
    // 根据 k 值选择模板
}

