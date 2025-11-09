/*
 * 独立的 Top-K Kernel 实现
 * 整合自 Faiss GPU Top-K 逻辑，无外部依赖
 */

#include <cuda_runtime.h>
#include <limits>

// ============================================================================
// 基础工具函数
// ============================================================================

__device__ __forceinline__ int getLaneId() {
    return threadIdx.x % 32;
}

__device__ __forceinline__ int getWarpId() {
    return threadIdx.x / 32;
}

template<typename T>
__device__ __forceinline__ T shfl_xor(T val, int laneMask) {
    return __shfl_xor_sync(0xffffffff, val, laneMask);
}

template<typename T>
__device__ __forceinline__ T shfl(T val, int srcLane) {
    return __shfl_sync(0xffffffff, val, srcLane);
}

// ============================================================================
// 比较器
// ============================================================================

template<typename T>
struct Comparator {
    __device__ static inline bool gt(T a, T b) { return a > b; }
    __device__ static inline bool lt(T a, T b) { return a < b; }
};

template<typename T>
struct Limits {
    __device__ static inline T getMin() { return -std::numeric_limits<T>::infinity(); }
    __device__ static inline T getMax() { return std::numeric_limits<T>::infinity(); }
};

// ============================================================================
// Warp 级 Bitonic Sort（用于小规模排序）
// ============================================================================

template<typename K, typename V, bool Dir>
__device__ inline void bitonicSwap(K& k, V& v, int mask, int dir_mask) {
    K other_k = shfl_xor(k, mask);
    V other_v = shfl_xor(v, mask);
    
    int lane_id = getLaneId();
    bool swap = ((lane_id & dir_mask) == 0) ? 
                (Dir ? (k < other_k) : (k > other_k)) :
                (Dir ? (k > other_k) : (k < other_k));
    
    if (swap) {
        k = other_k;
        v = other_v;
    }
}

template<typename K, typename V, int N, bool Dir>
__device__ inline void warpBitonicSort(K keys[N], V vals[N]) {
    #pragma unroll
    for (int i = 0; i < N; i++) {
        K k = keys[i];
        V v = vals[i];
        
        // Bitonic sort network
        #pragma unroll
        for (int size = 2; size <= 32; size *= 2) {
            int dir_mask = size / 2;
            
            #pragma unroll
            for (int stride = size / 2; stride > 0; stride /= 2) {
                bitonicSwap<K, V, Dir>(k, v, stride, dir_mask);
            }
        }
        
        keys[i] = k;
        vals[i] = v;
    }
}

// ============================================================================
// Warp 级 Merge（合并两个已排序的寄存器数组）
// ============================================================================

template<typename K, typename V, int N1, int N2, bool Dir>
__device__ inline void warpMerge(
    K keys1[N1], V vals1[N1],
    const K keys2[N2], const V vals2[N2]
) {
    // 简化版：将两个数组合并，然后排序
    // 实际 Faiss 使用更复杂的 merge network
    
    K temp_k[N1 + N2];
    V temp_v[N1 + N2];
    
    #pragma unroll
    for (int i = 0; i < N1; i++) {
        temp_k[i] = keys1[i];
        temp_v[i] = vals1[i];
    }
    
    #pragma unroll
    for (int i = 0; i < N2; i++) {
        temp_k[N1 + i] = keys2[i];
        temp_v[N1 + i] = vals2[i];
    }
    
    // 冒泡排序（小数组，编译时展开）
    #pragma unroll
    for (int i = 0; i < N1; i++) {
        #pragma unroll
        for (int j = i + 1; j < N1 + N2; j++) {
            bool swap = Dir ? (temp_k[i] < temp_k[j]) : (temp_k[i] > temp_k[j]);
            if (swap) {
                K tk = temp_k[i]; temp_k[i] = temp_k[j]; temp_k[j] = tk;
                V tv = temp_v[i]; temp_v[i] = temp_v[j]; temp_v[j] = tv;
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < N1; i++) {
        keys1[i] = temp_k[i];
        vals1[i] = temp_v[i];
    }
}

// ============================================================================
// WarpSelect: Warp 级别的 Top-K 选择
// ============================================================================

template<
    typename K,           // Key 类型（例如 float）
    typename V,           // Value 类型（例如 int）
    bool Dir,             // true=最大值, false=最小值
    int NumWarpQ,         // Warp 队列大小（必须是 32 的倍数）
    int NumThreadQ        // 每个线程的队列大小
>
struct WarpSelect {
    static constexpr int kNumWarpQRegisters = NumWarpQ / 32;
    
    // 每个线程的队列（寄存器）
    K threadK[NumThreadQ];
    V threadV[NumThreadQ];
    
    // Warp 队列（shared memory，每个 lane 持有一部分）
    K warpK[kNumWarpQRegisters];
    V warpV[kNumWarpQRegisters];
    
    K initK;
    V initV;
    int numVals;  // threadQ 中有效元素数量
    K warpKTop;   // 当前 Top-K 的阈值
    
    __device__ inline WarpSelect(K initKVal, V initVVal, int k)
        : initK(initKVal), initV(initVVal), numVals(0), warpKTop(initKVal) {
        
        // 初始化 thread queue
        #pragma unroll
        for (int i = 0; i < NumThreadQ; ++i) {
            threadK[i] = initK;
            threadV[i] = initV;
        }
        
        // 初始化 warp queue
        #pragma unroll
        for (int i = 0; i < kNumWarpQRegisters; ++i) {
            warpK[i] = initK;
            warpV[i] = initV;
        }
    }
    
    // 添加元素到 thread queue
    __device__ inline void addThreadQ(K k, V v) {
        if (Dir ? (k > warpKTop) : (k < warpKTop)) {
            // 插入排序（右移）
            #pragma unroll
            for (int i = NumThreadQ - 1; i > 0; --i) {
                threadK[i] = threadK[i - 1];
                threadV[i] = threadV[i - 1];
            }
            
            threadK[0] = k;
            threadV[0] = v;
            ++numVals;
        }
    }
    
    // 检查并合并 thread queue 到 warp queue
    __device__ inline void checkThreadQ() {
        bool needSort = (numVals == NumThreadQ);
        needSort = __any_sync(0xffffffff, needSort);
        
        if (!needSort) return;
        
        // 合并到 warp queue
        mergeWarpQ();
        
        // 重置 thread queue
        numVals = 0;
        #pragma unroll
        for (int i = 0; i < NumThreadQ; ++i) {
            threadK[i] = initK;
            threadV[i] = initV;
        }
        
        // 更新阈值
        int kLane = (NumWarpQ - 1) % 32;
        warpKTop = shfl(warpK[kNumWarpQRegisters - 1], kLane);
    }
    
    // 合并 thread queue 和 warp queue
    __device__ inline void mergeWarpQ() {
        int lane_id = getLaneId();
        
        // 排序 thread queue
        warpBitonicSort<K, V, NumThreadQ, !Dir>(threadK, threadV);
        
        // 从 shared memory 风格访问 warp queue
        K warpKRegs[kNumWarpQRegisters];
        V warpVRegs[kNumWarpQRegisters];
        
        #pragma unroll
        for (int i = 0; i < kNumWarpQRegisters; ++i) {
            warpKRegs[i] = warpK[i];
            warpVRegs[i] = warpV[i];
        }
        
        // 合并两个已排序列表
        warpMerge<K, V, kNumWarpQRegisters, NumThreadQ, !Dir>(
            warpKRegs, warpVRegs, threadK, threadV
        );
        
        // 写回
        #pragma unroll
        for (int i = 0; i < kNumWarpQRegisters; ++i) {
            warpK[i] = warpKRegs[i];
            warpV[i] = warpVRegs[i];
        }
    }
    
    // 添加元素（组合操作）
    __device__ inline void add(K k, V v) {
        addThreadQ(k, v);
        checkThreadQ();
    }
    
    // 最终归约
    __device__ inline void reduce() {
        mergeWarpQ();
    }
    
    // 写出结果
    __device__ inline void writeOut(K* outK, V* outV, int k) {
        int lane_id = getLaneId();
        
        #pragma unroll
        for (int i = 0; i < kNumWarpQRegisters; ++i) {
            int idx = i * 32 + lane_id;
            if (idx < k) {
                outK[idx] = warpK[i];
                outV[idx] = warpV[i];
            }
        }
    }
};

// ============================================================================
// BlockSelect: Block 级别的 Top-K 选择
// ============================================================================

template<
    typename K,
    typename V,
    bool Dir,
    int NumWarpQ,
    int NumThreadQ,
    int ThreadsPerBlock
>
struct BlockSelect {
    static constexpr int kNumWarps = ThreadsPerBlock / 32;
    
    K* sharedK;
    V* sharedV;
    
    K* warpK;
    V* warpV;
    
    WarpSelect<K, V, Dir, NumWarpQ, NumThreadQ> warpSelect;
    
    __device__ inline BlockSelect(
        K initK, V initV,
        K* smemK, V* smemV,
        int k
    ) : sharedK(smemK),
        sharedV(smemV),
        warpSelect(initK, initV, k) {
        
        int warp_id = getWarpId();
        warpK = &sharedK[warp_id * NumWarpQ];
        warpV = &sharedV[warp_id * NumWarpQ];
    }
    
    __device__ inline void add(K k, V v) {
        warpSelect.add(k, v);
    }
    
    __device__ inline void addThreadQ(K k, V v) {
        warpSelect.addThreadQ(k, v);
    }
    
    __device__ inline void reduce() {
        warpSelect.reduce();
        
        // 写入 shared memory
        int lane_id = getLaneId();
        if (lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < NumWarpQ; i++) {
                warpK[i] = warpSelect.warpK[i / 32];
                warpV[i] = warpSelect.warpV[i / 32];
            }
        }
        
        __syncthreads();
        
        // Block-wide merge（在 warp 0 中进行）
        if (getWarpId() == 0) {
            blockWiseMerge(sharedK, sharedV);
        }
        
        __syncthreads();
    }
    
private:
    // Block 级合并（递归合并所有 warp 的结果）
    __device__ inline void blockWiseMerge(K* smemK, V* smemV) {
        int lane_id = getLaneId();
        
        // 简化版：warp 0 收集所有 warp 的 Top-K 并重新选择
        WarpSelect<K, V, Dir, NumWarpQ, NumThreadQ> final_select(
            warpSelect.initK, warpSelect.initV, NumWarpQ
        );
        
        // 从所有 warp 收集
        for (int w = 0; w < kNumWarps; w++) {
            for (int i = lane_id; i < NumWarpQ; i += 32) {
                K k = smemK[w * NumWarpQ + i];
                V v = smemV[w * NumWarpQ + i];
                final_select.add(k, v);
            }
        }
        
        final_select.reduce();
        
        // 写回 shared memory 的前 NumWarpQ 个位置
        #pragma unroll
        for (int i = 0; i < NumWarpQ / 32; i++) {
            int idx = i * 32 + lane_id;
            if (idx < NumWarpQ) {
                smemK[idx] = final_select.warpK[i];
                smemV[idx] = final_select.warpV[i];
            }
        }
    }
};

// ============================================================================
// Top-K Selection Kernel
// ============================================================================

/*
 * 通用 Top-K 选择 kernel
 * 
 * 参数：
 *   in: [num_rows, num_cols] 输入矩阵
 *   outK: [num_rows, k] 输出 Top-K 值
 *   outV: [num_rows, k] 输出 Top-K 索引
 *   k: Top-K 数量
 *   dir: true=最大值, false=最小值
 */
template<
    typename K,
    typename V,
    bool Dir,
    int NumWarpQ,
    int NumThreadQ,
    int ThreadsPerBlock
>
__global__ void topKSelectKernel(
    const K* __restrict__ in,     // [num_rows × num_cols]
    K* __restrict__ outK,          // [num_rows × k]
    V* __restrict__ outV,          // [num_rows × k]
    int num_rows,
    int num_cols,
    int k
) {
    constexpr int kNumWarps = ThreadsPerBlock / 32;
    
    // Shared memory for block-wide Top-K
    __shared__ K smemK[kNumWarps * NumWarpQ];
    __shared__ V smemV[kNumWarps * NumWarpQ];
    
    // 每个 block 处理一行
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    K initK = Dir ? Limits<K>::getMin() : Limits<K>::getMax();
    V initV = -1;
    
    BlockSelect<K, V, Dir, NumWarpQ, NumThreadQ, ThreadsPerBlock> 
        heap(initK, initV, smemK, smemV, k);
    
    // 遍历这一行的所有元素
    const K* row_data = in + row * num_cols;
    
    int tid = threadIdx.x;
    int i = tid;
    
    // Warp-aligned 处理
    int limit = (num_cols / 32) * 32;
    
    for (; i < limit; i += ThreadsPerBlock) {
        heap.add(row_data[i], (V)i);
    }
    
    // 处理剩余元素
    if (i < num_cols) {
        heap.addThreadQ(row_data[i], (V)i);
    }
    
    // 归约得到 Top-K
    heap.reduce();
    
    // 写出结果
    for (int i = tid; i < k; i += ThreadsPerBlock) {
        outK[row * k + i] = smemK[i];
        outV[row * k + i] = smemV[i];
    }
}

// ============================================================================
// 简化版 Top-K Kernel（更容易理解和修改）
// ============================================================================

template<int K>
struct SimpleTopK {
    // 降序排列，values[0] 最大，values[K-1] 最小
    float values[K];
    // values中对应数据在原始数据中的col index
    int indices[K];
    
    __device__ inline SimpleTopK() {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            values[i] = -INFINITY;
            indices[i] = -1;
        }
    }
    
    // 添加元素到 Top-K
    // 这个函数的
    // 这个函数的作用是，将新元素的val和col idx插入values和indices的对应位置
    // 同时values和indices中的其余元素整体右移一位
    // 这里需要注意的是，当一行的数据个数小于K是，会存在values和indices填不满的情况，values和indices中会存在默认填充值
    __device__ inline void add(float val, int idx) {
        // 早期退出：比最小的还小
        if (val <= values[K-1]) return;
        
        // 找到插入位置
        int pos = K - 1;
        #pragma unroll
        for (int i = 0; i < K; i++) {
            if (val > values[i]) {
                pos = i;
                break;
            }
        }
        
        // 插入：右移元素
        // 把values和indices中的[pos+1, K)的元素整体右移一位
        #pragma unroll
        for (int i = K - 1; i > pos; i--) {
            values[i] = values[i - 1];
            indices[i] = indices[i - 1];
        }
        // 新元素插入到values[pos]和indices[pos]
        values[pos] = val;
        indices[pos] = idx;
    }
    
    // Warp 内归约（合并 32 个线程的 Top-K）
    __device__ inline void warpReduce() {
        int lane_id = getLaneId();
        
        // 使用 warp shuffle 收集其他线程的数据
        #pragma unroll
        // 这里的warp shuffle可以参考leetcuda的softmax.cu里面的warp_reduce_max_f32
        // 一个warp中每个线程存储了K个最大值，相当于一个warp中总共存储了32*K个最大值
        // 执行完下面两个for循环之后，32*K个值中最大的K个值就会被存放在每个warp的lane_id为0的线程中
        for(int offset = 32 >> 1; offset >= 1; offset >>= 1) {
            #pragma unroll
            // 遍历 values[K]和indices[K]中所有元素
            for (int i = 0; i < K; i++) {
                float other_val = shfl_xor(values[i], offset);
                int other_idx = shfl_xor(indices[i], offset);
                
                // 这里排除边界情况，当原始矩阵一行的数据数量小于K时，values和indices会存在默认填充值的情况
                // 这里的判断就是为了排除默认填充值
                if (other_idx >= 0) {
                    add(other_val, other_idx);
                }
            }
        }
    }
    
    // Block 内归约（合并所有 warp 的 Top-K）
    __device__ inline void blockReduce(
        float* smem_vals,
        int* smem_idxs,
        int num_warps
    ) {
        int warp_id = getWarpId();
        int lane_id = getLaneId();
        
        // Lane 0 写入 shared memory
        // 每个warp的lane 0把结果写到shared mem里面去
        // 执行完下面for循环之后，一个block中的所有warp的lane0的结果都在smem中了
        if (lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                smem_vals[warp_id * K + i] = values[i];
                smem_idxs[warp_id * K + i] = indices[i];
            }
        }
        __syncthreads();
        
        // Warp 0 的lane 0 来做最终规约，对smem中的数据全部都调用一遍add方法，最终的topk的结果就存放在warp0的lane0中
        if (warp_id == 0 && lane_id == 0) {
            // 重置自己的队列
            #pragma unroll
            for (int i = 0; i < K; i++) {
                values[i] = -INFINITY;
                indices[i] = -1;
            }
            
            // 从所有 warp 收集
            for (int w = 0; w < num_warps; w++) {
                #pragma unroll
                for (int i = 0; i < K; i++) {
                    float val = smem_vals[w * K + i];
                    int idx = smem_idxs[w * K + i];
                    // 与add方法一样的边界条件处理
                    if (idx >= 0) {
                        add(val, idx);
                    }
                }
            }
        }
        
        __syncthreads();
    }
};

// ============================================================================
// 简化的 Top-K Selection Kernel
// ============================================================================

template<int K>
__global__ void simpleTopKKernel(
    const float* __restrict__ input,   // [num_rows × num_cols]
    float* __restrict__ outK,          // [num_rows × k]
    int* __restrict__ outV,            // [num_rows × k]
    int num_rows,
    int num_cols,
    int k
) {
    // 每个 block 处理一行
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    const int tid = threadIdx.x;
    constexpr int BLOCK_SIZE = 256;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    
    // 每个线程维护 Top-K
    // 这里面维护了一个values[K]和indices[K]
    // 其中values降序排列，values[0] 最大，values[K-1] 最小
    // indices中存储了values中对应数据在原始数据中的col index
    SimpleTopK<K> topk;
    
    // 遍历这一行
    const float* row_data = input + row * num_cols;
    
    // 这里以整个block的视角来考虑，在这个例子中，一个block有4个warp
    // 4个warp执行完了这一行代码之后，相当于原始的矩阵中一行的num_cols的元素都被add方法处理过了
    // 执行完成之后，每个线程的values和indices里面都存放了当前线程所遍历的结果里面的topk的结果的val和col idx
    for (int col = tid; col < num_cols; col += BLOCK_SIZE) {
        topk.add(row_data[col], col);
    }
    
    // Warp reduce
    // 一个warp中每个线程存储了K个最大值，相当于一个warp中总共存储了32*K个最大值
    // warpReduce后，32*K个值中最大的K个值就会被存放在每个warp的lane_id为0的线程中
    topk.warpReduce();
    
    // Block reduce
    __shared__ float smem_vals[NUM_WARPS * K];
    __shared__ int smem_idxs[NUM_WARPS * K];
    
    topk.blockReduce(smem_vals, smem_idxs, NUM_WARPS);
    
    // 写出结果（只有 warp 0 的 lane 0）
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < k; i++) {
            outK[row * k + i] = topk.values[i];
            outV[row * k + i] = topk.indices[i];
        }
    }
}

// ============================================================================
// Host 接口函数
// ============================================================================

extern "C" void launchTopKKernel(
    const float* d_input,
    float* d_outK,
    int* d_outV,
    int num_rows,
    int num_cols,
    int k,
    bool select_max,
    cudaStream_t stream
) {
    dim3 grid(num_rows);
    dim3 block(256);
    
    // 根据 k 值选择合适的模板实例
    if (k <= 10) {
        simpleTopKKernel<10><<<grid, block, 0, stream>>>(
            d_input, d_outK, d_outV, num_rows, num_cols, k
        );
    } else if (k <= 32) {
        simpleTopKKernel<32><<<grid, block, 0, stream>>>(
            d_input, d_outK, d_outV, num_rows, num_cols, k
        );
    } else if (k <= 64) {
        simpleTopKKernel<64><<<grid, block, 0, stream>>>(
            d_input, d_outK, d_outV, num_rows, num_cols, k
        );
    } else if (k <= 128) {
        simpleTopKKernel<128><<<grid, block, 0, stream>>>(
            d_input, d_outK, d_outV, num_rows, num_cols, k
        );
    } else if (k <= 256) {
        simpleTopKKernel<256><<<grid, block, 0, stream>>>(
            d_input, d_outK, d_outV, num_rows, num_cols, k
        );
    } else if (k <= 512) {
        simpleTopKKernel<512><<<grid, block, 0, stream>>>(
            d_input, d_outK, d_outV, num_rows, num_cols, k
        );
    } else if (k <= 1024) {
        simpleTopKKernel<1024><<<grid, block, 0, stream>>>(
            d_input, d_outK, d_outV, num_rows, num_cols, k
        );
    } else if (k <= 2048) {
        simpleTopKKernel<2048><<<grid, block, 0, stream>>>(
            d_input, d_outK, d_outV, num_rows, num_cols, k
        );
    }
}

