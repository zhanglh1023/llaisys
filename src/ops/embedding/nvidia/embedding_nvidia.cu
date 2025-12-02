#include "llaisys.h"
#include "../../../core/llaisys_core.hpp"
#include "../../../device/nvidia/utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

template <typename T>
__global__ void embedding_(T *out,
                            const int64_t *index,
                            const T *weight,
                            size_t hidden_size,
                            size_t numel) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int b_sz = blockDim.x;
    int off_set = bx * hidden_size;
    for(size_t i = 0;i < hidden_size;i += b_sz) {
        if(i + tx >= hidden_size) continue;
        int w_off_set = index[bx] * hidden_size;
        out[off_set + i + tx] = weight[w_off_set + i + tx];
    }
}


} // anomynous namespace

namespace llaisys::ops::nvidia {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, const size_t numel,
                const std::vector<size_t>& shape, const std::vector<int64_t>& strides) {
    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    dim3 block(256);
    dim3 grid(numel);
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        embedding_<float><<<grid, block, 0, s>>>(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), 
                    reinterpret_cast<const float *>(weight), shape[1], numel);
        break;
    }
    case LLAISYS_DTYPE_BF16:{
        embedding_<__nv_bfloat16><<<grid, block, 0, s>>>(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const int64_t *>(index), 
                    reinterpret_cast<const __nv_bfloat16 *>(weight), shape[1], numel);
        break;
    }
    case LLAISYS_DTYPE_F16:
    {
        embedding_<__half><<<grid, block, 0, s>>>(reinterpret_cast<__half *>(out), reinterpret_cast<const int64_t *>(index), 
                    reinterpret_cast<const __half *>(weight), shape[1], numel);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // llaisys::ops::nvidia