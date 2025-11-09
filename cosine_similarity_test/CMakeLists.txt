cmake_minimum_required(VERSION 3.18)
project(FaissCosineTest CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 添加 OpenMP（在使用前查找）
find_package(OpenMP REQUIRED)

# 查找 CUDA
find_package(CUDAToolkit REQUIRED)

# 查找 BLAS (Faiss 依赖)
find_package(BLAS REQUIRED)

# Faiss 路径配置
set(FAISS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Faiss root directory")
set(FAISS_BUILD_DIR "${FAISS_ROOT}/build" CACHE PATH "Faiss build directory")

# 检查 Faiss 是否已编译
if(NOT EXISTS "${FAISS_BUILD_DIR}/faiss/libfaiss.so" AND NOT EXISTS "${FAISS_BUILD_DIR}/faiss/libfaiss.a")
    message(WARNING "Faiss library not found in ${FAISS_BUILD_DIR}/faiss")
    message(WARNING "Please compile Faiss first with GPU support enabled")
    message(WARNING "See instructions below in this CMakeLists.txt")
endif()

# 包含目录
include_directories(
    ${FAISS_ROOT}
    ${FAISS_BUILD_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 链接目录
link_directories(
    ${FAISS_BUILD_DIR}/faiss
    ${CUDAToolkit_LIBRARY_DIR}
)

# 可执行文件
add_executable(main_test 
    main_test.cpp
    gpu_normalize.cu
)

# 独立的 Top-K 测试程序
add_executable(test_topk
    test_topk.cpp
    topk_kernel.cu
)

# 设置 CUDA 架构（根据你的 GPU 修改）
set_target_properties(main_test PROPERTIES 
    CUDA_ARCHITECTURES "75;80;86;89;90"  # 支持 Turing, Ampere, Ada, Hopper
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(test_topk PROPERTIES 
    CUDA_ARCHITECTURES "75;80;86;89;90"
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接库
target_link_libraries(main_test PRIVATE
    faiss
    CUDA::cudart
    CUDA::cublas
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
    pthread
    dl
)

# test_topk 不需要 Faiss 库（独立实现）
target_link_libraries(test_topk PRIVATE
    CUDA::cudart
)

# 编译选项
target_compile_options(main_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda>
)

target_compile_options(test_topk PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda>
)

# 打印配置信息
message(STATUS "==================== Configuration ====================")
message(STATUS "FAISS_ROOT: ${FAISS_ROOT}")
message(STATUS "FAISS_BUILD_DIR: ${FAISS_BUILD_DIR}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "=======================================================")