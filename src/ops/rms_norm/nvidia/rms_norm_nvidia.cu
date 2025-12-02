#include "llaisys.h"
#include "../../ops.hpp"
#include "../../../device/nvidia/utils.cuh"


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>



namespace {
#define WARP_SIZE 32
#define CEIL(a, b) (((a) + (b) - 1) / (b))
__device__ __forceinline__ float warp_reduce_sum_f32(float value) {
    #pragma unroll
    for(size_t i = WARP_SIZE >> 1;i > 0;i >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, i);
    }
    return value;
}
template<const int WARP_NUM = 32>
__device__ __forceinline__ float block_reduce_sum_f32(float value) {
    value = warp_reduce_sum_f32(value);
    static __shared__ float s_mem[WARP_SIZE];
    int laneid = threadIdx.x % WARP_SIZE;
    int warpid = threadIdx.x / WARP_SIZE;
    if(laneid == 0) {
        s_mem[warpid] = value;
    }
    __syncthreads();
    if(warpid == 0) {
        value = (laneid < WARP_NUM) ? s_mem[laneid] : 0.f;
        value = warp_reduce_sum_f32(value);
        if(laneid == 0) return value;
    }
}
template<typename T, const int THREAD_NUM = 256>
__global__ void rms_norm_(T* out, const T* in, const T* weight, const float eps, const size_t N, const size_t K) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    in += bx * K;
    out += bx * K;
    float value = 0.f;
    #pragma unroll
    for(size_t i = 0;i < K;i += THREAD_NUM) {
        float x = static_cast<float>(in[i + tx]);
        if(tx + i < K) value += x * x;
    }
    value = block_reduce_sum_f32<CEIL(THREAD_NUM, WARP_SIZE)>(value);
    __shared__ T s_value;
    if(tx == 0) {
        s_value = static_cast<T>(rsqrtf(value / K + eps));
    }
    __syncthreads();
    #pragma unroll
    for(size_t i = 0;i < K;i += THREAD_NUM) {
        if(tx + i < K) out[i + tx] = in[i + tx] * s_value * weight[i + tx];
    }
}
}


namespace llaisys::ops::nvidia {

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
            const size_t N, const size_t K, const float eps,
            llaisysDataType_t type) {

    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    dim3 block(256);
    dim3 grid(N);
    switch (type) {
		case LLAISYS_DTYPE_F32:
			rms_norm_<float><<<grid, block, 0, s>>>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), reinterpret_cast<const float*>(weight),
                                                eps, N, K);
			break;
		case LLAISYS_DTYPE_F16:
			rms_norm_<__half><<<grid, block, 0, s>>>(reinterpret_cast<__half*>(out), reinterpret_cast<const __half*>(in), reinterpret_cast<const __half*>(weight),
                                                eps, N, K);
			break;
		case LLAISYS_DTYPE_BF16:
			rms_norm_<__nv_bfloat16><<<grid, block, 0, s>>>(reinterpret_cast<__nv_bfloat16*>(out), reinterpret_cast<const __nv_bfloat16*>(in), reinterpret_cast<const __nv_bfloat16*>(weight),
                                                eps, N, K);
            break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}