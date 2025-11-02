# 编译和运行余弦相似度测试

本文档说明如何编译 Faiss 和余弦相似度测试程序。

## 前置条件

1. CUDA Toolkit (>= 11.0)
2. CMake (>= 3.18)
3. GCC/G++ (>= 7.0)
4. OpenMP
5. BLAS 库 (OpenBLAS 或 Intel MKL)

## 步骤 1: 编译 Faiss (首次运行必须)

如果你还没有编译 Faiss，需要先编译 Faiss 库（**启用 GPU 支持**）：

```bash
# 进入 Faiss 根目录
cd /home/p/Workspace/code/cuda_learn/faiss

# 创建构建目录
mkdir -p build
cd build

# 配置 CMake (启用 GPU 支持)
cmake .. \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
    -DBUILD_TESTING=OFF

# 编译 (使用多线程加速)
make -j$(nproc)

# 验证编译结果
ls faiss/libfaiss.so  # 应该看到这个文件
```

**注意**: 
- 如果你的 GPU 是特定架构，可以只编译对应的架构以加快编译速度
- 常见架构: `75` (Turing), `80` (A100), `86` (RTX 3090), `89` (RTX 4090), `90` (H100)
- 查看你的 GPU 架构: `nvidia-smi --query-gpu=compute_cap --format=csv`

## 步骤 2: 编译余弦相似度测试程序

```bash
# 进入测试目录
cd /home/p/Workspace/code/cuda_learn/faiss/cosine_similarity_test

# 创建构建目录
mkdir -p build
cd build

# 配置 CMake
cmake ..

# 编译
make -j$(nproc)

# 验证可执行文件
ls main_test  # 应该看到这个可执行文件
```

## 步骤 3: 运行测试程序

```bash
# 在 build 目录下运行
./main_test
```

**预期输出**:
```
余弦相似度计算 - 直接调用 CUDA Kernel
======================================================================
参数配置:
  向量维度: 128
  数据库大小: 10000
  查询数量: 100
  Top-K: 10
======================================================================

[步骤 1] 在 CPU 上生成测试数据...
  ✓ 生成 10000 个数据库向量
  ✓ 生成 100 个查询向量

[步骤 2] 使用原生 CUDA API 分配 GPU 内存...
  ✓ cudaMalloc d_database: 4.88 MB
  ✓ cudaMalloc d_queries: 0.05 MB
  ✓ cudaMalloc d_distances: 3.81 MB

[步骤 3] 使用 cudaMemcpy 拷贝数据到 GPU...
  ✓ 数据拷贝完成 (耗时: X.XX ms)

[步骤 4] GPU 端 L2 归一化...
  直接调用 Faiss 的 l2NormRowMajor __global__ kernel
  ✓ GPU 归一化完成 (耗时: X.XX ms)
  验证: 第一个向量范数 = 1.000000 (应接近 1.0)

[步骤 5] 创建 cuBLAS handle 并直接调用 GEMM kernel...
  调用 cublasSgemm 计算内积 (这是 Faiss 内部使用的 kernel):
    d_distances = d_queries × d_database^T
    矩阵维度: [100 × 128] × [128 × 10000]
  ✓ GEMM 计算完成 (耗时: X.XX ms)
  性能: XXXX.XX GFLOPS

...
```

## 一键脚本（推荐）

创建一个一键编译运行的脚本:

```bash
#!/bin/bash
# 保存为 build_and_run.sh

set -e  # 遇到错误立即退出

echo "========================================"
echo "Step 1: 检查 Faiss 是否已编译"
echo "========================================"

FAISS_ROOT="/home/p/Workspace/code/cuda_learn/faiss"
FAISS_BUILD="${FAISS_ROOT}/build"

if [ ! -f "${FAISS_BUILD}/faiss/libfaiss.so" ]; then
    echo "Faiss 未编译，开始编译 Faiss..."
    cd ${FAISS_ROOT}
    mkdir -p build && cd build
    cmake .. \
        -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF
    make -j$(nproc)
    echo "✓ Faiss 编译完成"
else
    echo "✓ Faiss 已编译"
fi

echo ""
echo "========================================"
echo "Step 2: 编译余弦相似度测试程序"
echo "========================================"

cd ${FAISS_ROOT}/cosine_similarity_test
mkdir -p build && cd build
cmake ..
make -j$(nproc)

echo ""
echo "========================================"
echo "Step 3: 运行测试程序"
echo "========================================"

./main_test

echo ""
echo "✓ 全部完成！"
```

然后执行:
```bash
chmod +x build_and_run.sh
./build_and_run.sh
```

## 常见问题

### 1. CUDA 架构不匹配
```
error: unsupported GPU architecture 'compute_XX'
```
**解决**: 在 CMakeLists.txt 中修改 `CUDA_ARCHITECTURES` 为你的 GPU 架构。

### 2. 找不到 libfaiss.so
```
error while loading shared libraries: libfaiss.so
```
**解决**: 设置 LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=/home/p/Workspace/code/cuda_learn/faiss/build/faiss:$LD_LIBRARY_PATH
./main_test
```

### 3. 找不到 OpenBLAS 或 MKL
```
Could not find a package configuration file provided by "OpenBLAS"
```
**解决**: 安装 BLAS 库:
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# 或使用 Intel MKL
# 下载并安装 Intel oneAPI Math Kernel Library
```

### 4. 编译 Faiss 时提示 CUDA 相关错误
**解决**: 确保 CUDA 环境变量设置正确:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## 性能调优提示

1. **针对特定 GPU 架构编译**: 只编译你的 GPU 架构可以获得更好的性能
2. **使用 Release 模式**: 确保使用 `-DCMAKE_BUILD_TYPE=Release`
3. **启用 Tensor Cores** (Volta+): 在代码中使用混合精度 (FP16/FP32)
4. **增加数据规模**: 更大的 `nb` 和 `nq` 可以更好地发挥 GPU 性能

## 测试不同参数

修改 `main_test.cpp` 中的参数:
```cpp
const int d = 128;      // 向量维度
const int nb = 10000;   // 数据库大小
const int nq = 100;     // 查询数量
const int k = 10;       // Top-K
```

重新编译运行:
```bash
cd build
make
./main_test
```

