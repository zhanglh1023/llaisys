#include "llaisys.h"
#include "../../ops.hpp"
#include "../../../device/nvidia/utils.cuh"


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

namespace {

template <typename T>
__device__ __forceinline__ T add(float a, T b) {
    return a + b;
}
template <>
__device__ __forceinline__ __half add(float a, __half b) {
    __half ta = __float2half(a);
    return __hadd(ta, b);
}
template <>
__device__ __forceinline__ __nv_bfloat16 add(float a, __nv_bfloat16 b) {
    __nv_bfloat16 ta = __float2bfloat16(a);
    return __hadd(ta, b);
}
template <typename T>
__device__ __forceinline__ float fma(T a, T b, float c) {
    return a * b + c;
}
template <>
__device__ __forceinline__ float fma(__half a, __half b, float c) {
    float ta = __half2float(a);
    float tb = __half2float(b);
    return ta * tb + c;
}
template <>
__device__ __forceinline__ float fma(__nv_bfloat16 a, __nv_bfloat16 b, float c) {
    float ta = __bfloat162float(a);
    float tb = __bfloat162float(b);
    return ta * tb + c;
}

template <typename T, const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm(T* C, const T* A, const T* B, const T* BIAS, const size_t M, const size_t N, const size_t K) {
    extern __shared__ char s_mem[];
    T *s_a = reinterpret_cast<T*>(s_mem);
    T *s_b = s_a + BM * BK;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_col = BN / TN;
    int thread_row = BM / TM;
    int thread_num = thread_row * thread_col;
    int tx = (threadIdx.x % thread_col) * TN;
    int ty = (threadIdx.x / thread_col) * TM;

    A += by * BM * K;
    B += bx * BN * K;
    C += by * BM * N + bx * BN;

    int a_col = threadIdx.x % BK;
    int a_row = threadIdx.x / BK;
    int a_stride = thread_num / BK;

    int b_col = threadIdx.x % BK;
    int b_row = threadIdx.x / BK;
    int b_stride = thread_num / BK;

    float tmp[TM][TN] = {0.f};
    
    for(size_t bk = 0;bk < K;bk += BK) {
        for(size_t i = 0;i < BM;i += a_stride) {
            s_a[(a_row + i) * BK + a_col] = (by * BM + a_row + i < M && a_col + bk < K) ? A[(a_row + i) * K + a_col] : static_cast<T>(0.0);
        }
        for(size_t i = 0;i < BN;i += b_stride) {
            s_b[(b_row + i) * BK + b_col] = (bx * BN + b_row + i < N && b_col + bk < K) ? B[(b_row + i) * K + b_col] : static_cast<T>(0.0);
        }
        __syncthreads();
        if(bk + BK < K) {
            A += BK;
            B += BK;
        }
        
        for(size_t k = 0;k < BK;k++) {
            for(size_t i = 0;i < TM;i++) {
                for(size_t j = 0;j < TN;j++) {
                    tmp[i][j] = fma(s_a[(ty + i) * BK + k], s_b[(tx + j) * BK + k], tmp[i][j]);
                    //tmp[i][j] += s_a[(ty + i) * BK + k] * s_b[(tx + j) * BK + k];
                }
            }
        }
        __syncthreads();
    }

    for(int j = 0;j < TN;j++) {
        T bias = BIAS[bx * BN + tx + j];
        for(int i = 0;i < TM;i++) {
            if(by * BM + ty + i < M && bx * BN + tx + j < N) 
                C[(ty + i) * N + tx + j] = add(tmp[i][j], bias);
                //C[(ty + i) * N + tx + j] = tmp[i][j] + BIAS[bx * BN + tx + j];
        }
    }
}

template<typename T>
__global__ void add_bias(T* out, const T* in, const T* bias, size_t M, size_t N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= M * N) return ;
    int col = idx % N;
    out[idx] = in[idx] + bias[col];
}
template <typename T, cudaDataType_t CType>
void linear_(std::byte *out, // [m, n]  (row major)  /  [n, m] (col major)
             const std::byte *in, // [m, k] (row major) / [k, m] (col major)
             const std::byte *weight, // [n, k] (row major) / [k, n] (col major)
             const std::byte *bias,
             size_t m,  // batch (rows of out)
             size_t k,  // in_dim
             size_t n,  // out_dim (cols of out)
             cudaStream_t stream) {
    /*
    if(m <= 256 && n <= 256 || k <= 256) {
        const int TM = 4;
        const int TN = 4;
        const int BM = 16 * TM;
        const int BN = 16 * TN;
        const int BK = 16;
        dim3 block(256);
        dim3 grid(CEIL(n, BN), CEIL(m, BM));
        gemm<T, BM, BN, BK, TM, TN><<<grid, block, 2 * BM * BN * sizeof(T), stream>>>(reinterpret_cast<T*>(out), reinterpret_cast<const T*>(in), reinterpret_cast<const T*>(weight),
                                                    reinterpret_cast<const T*>(bias), m, n, k);
    } else {*/
        auto &runtime = llaisys::core::context().runtime();
        auto handle = runtime.cublasHandle();

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        
        const auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

        // 数学目标（row 语义，不涉及存储）：
        //   out(m,n) = in(m,k) * weight^T(k,n)
        //
        // 关键等价（不拷贝、不物理转置）：
        //   (in * weight^T)^T = weight * in^T
        //
        // 存储等价（字节层面）：
        //   row(m,n) <=> col(n,m)
        //
        // 于是我们在 cuBLAS（列主序）里让它计算：
        //   C(n,m) = weight(n,k) * in^T(k,m) = out^T(n,m)
        //
        // 指针与参数映射（只改变“解释方式”，不做物理转置）：
        //   A <- weight  // row: [n,k]  → cuBLAS(col视角): [k,n],  lda = k,  opA = T  → (n,k)
        //   B <- in      // row: [m,k]  → cuBLAS(col视角): [k,m],  ldb = k,  opB = N  → (k,m) = in^T
        //   C <- out     // row: [m,n]  → cuBLAS(col视角): [n,m],  ldc = n
        //
        //   维度：m' = n,  n' = m,  k' = k
        //
        // opA/opB 是“数学转置标志”，用于纠正 cuBLAS 的列主序解释；不进行任何物理数据转置/拷贝。
        auto compute_type = std::is_same_v<T, __half> ? CUBLAS_COMPUTE_32F_FAST_16F : 
                        std::is_same_v<T, __nv_bfloat16> ? CUBLAS_COMPUTE_32F_FAST_16BF : CUBLAS_COMPUTE_32F;



        cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,                 // opA, opB
            static_cast<int>(n),                      // m' = n
            static_cast<int>(m),                      // n' = m
            static_cast<int>(k),                      // k
            &alpha,
            reinterpret_cast<const T *>(weight), CType, static_cast<int>(k), // A: lda = k
            reinterpret_cast<const T *>(in),     CType, static_cast<int>(k), // B: ldb = k
            &beta,
            reinterpret_cast<T *>(out),          CType, static_cast<int>(n), // C: ldc = n
            compute_type,
            algo);
        
        if(bias) {
            dim3 block(256);
            dim3 grid(CEIL(m * n, 256));
            add_bias<T><<<grid, block, 0, stream>>>(reinterpret_cast<T*>(out), reinterpret_cast<const T*>(out), reinterpret_cast<const T*>(bias), m, n);
        }
    //}

}


} // anomynous namespace


namespace llaisys::ops::nvidia {

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            size_t m, size_t k, size_t n,
            llaisysDataType_t type) {

    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    switch (type) {
		case LLAISYS_DTYPE_F32:
			linear_<float, CUDA_R_32F>(out, in, weight, bias,
									   m, k, n, s);
			break;
		case LLAISYS_DTYPE_F16:
			linear_<__half, CUDA_R_16F>(out, in, weight, bias,
										m, k, n, s);
			break;
		case LLAISYS_DTYPE_BF16:
			linear_<__nv_bfloat16, CUDA_R_16BF>(out, in, weight, bias,
												m, k, n, s);
			break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia