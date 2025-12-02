#include "llaisys.h"
#include "../../ops.hpp"
#include "../../../device/nvidia/utils.cuh"


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>



namespace {
#define CEIL(a, b) (((a) + (b) - 1) / (b))
template<typename T, const int THREAD_NUM>
__global__ void rope_(T* out, const T* in, const int64_t* pos_ids, const float theta, const size_t N) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    in += by * gridDim.x * N + bx * N;
    out += by * gridDim.x * N + bx * N;
    float pos_id = static_cast<float>(pos_ids[by]);
    #pragma unroll
    for(size_t i = 0;i < N / 2;i += THREAD_NUM) {
        int idx = i + tx;
        if(idx >= N / 2) return ;
        float phi = pos_id / powf(theta, idx * 2.0 / N);
        T sin_t = static_cast<T>(sinf(phi));
        T cos_t = static_cast<T>(cosf(phi));
        T a = in[idx];
        T b = in[idx + N / 2];
        out[idx] = a * cos_t - b * sin_t;
        out[idx + N / 2] = b * cos_t + a * sin_t;
    }
}
}


namespace llaisys::ops::nvidia {

void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
            const float theta, const size_t seq_len, const size_t nhead, const size_t dim,
            llaisysDataType_t type) {

    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    dim3 block(256);
    dim3 grid(nhead, seq_len);
    switch (type) {
		case LLAISYS_DTYPE_F32:
			rope_<float, 256><<<grid, block, 0, s>>>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), reinterpret_cast<const int64_t*>(pos_ids),
                                                theta, dim);
			break;
		case LLAISYS_DTYPE_F16:
			rope_<__half, 256><<<grid, block, 0, s>>>(reinterpret_cast<__half*>(out), reinterpret_cast<const __half*>(in), reinterpret_cast<const int64_t*>(pos_ids),
                                                theta, dim);
			break;
		case LLAISYS_DTYPE_BF16:
			rope_<__nv_bfloat16, 256><<<grid, block, 0, s>>>(reinterpret_cast<__nv_bfloat16*>(out), reinterpret_cast<const __nv_bfloat16*>(in), reinterpret_cast<const int64_t*>(pos_ids),
                                                theta, dim);
            break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}