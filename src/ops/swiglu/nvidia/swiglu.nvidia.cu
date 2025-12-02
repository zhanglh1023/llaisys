#include "llaisys.h"
#include "../../ops.hpp"
#include "../../../device/nvidia/utils.cuh"


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>



namespace {
#define WARP_SIZE 32
#define CEIL(a, b) (((a) + (b) - 1) / (b))

template<typename T, const int THREAD_NUM = 256>
__global__ void swiglu_(T* out, const T* gate, const T* up, const size_t N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N) return ;
    float t = static_cast<float>(gate[idx]);
    out[idx] = up[idx] * static_cast<T>(t / (1.f + expf(-t)));
}
}


namespace llaisys::ops::nvidia {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            const size_t N,
            llaisysDataType_t type) {

    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    dim3 block(256);
    dim3 grid(CEIL(N, 256));
    switch (type) {
		case LLAISYS_DTYPE_F32:
			swiglu_<float><<<grid, block, 0, s>>>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(gate), reinterpret_cast<const float*>(up),
                                                N);
			break;
		case LLAISYS_DTYPE_F16:
			swiglu_<__half><<<grid, block, 0, s>>>(reinterpret_cast<__half*>(out), reinterpret_cast<const __half*>(gate), reinterpret_cast<const __half*>(up),
                                                N);
			break;
		case LLAISYS_DTYPE_BF16:
			swiglu_<__nv_bfloat16><<<grid, block, 0, s>>>(reinterpret_cast<__nv_bfloat16*>(out), reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
                                                N);
            break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}