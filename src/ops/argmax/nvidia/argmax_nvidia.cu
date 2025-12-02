#include "argmax_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../ops.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
namespace {

template <typename T>
struct IndexValuePair {
    int64_t idx;
    T val;
};

template <typename T>
__device__ IndexValuePair<T> reduce_max_pair(IndexValuePair<T> a, IndexValuePair<T> b) {
    if(a.val > b.val) return a;
    return b;
}
template <typename T>
__global__ void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    extern __shared__ __align__(sizeof(IndexValuePair<T>))char shared_memory[];
    IndexValuePair<T>* s_mem = reinterpret_cast<IndexValuePair<T>*>(shared_memory);

    int idx = threadIdx.x;
    IndexValuePair<T> val = idx < n ? IndexValuePair<T>{idx, vals[idx]} : IndexValuePair<T>{0, vals[0]};

    for(size_t i = 0;i < n;i += blockDim.x) {
      if(i + threadIdx.x < n) {
        IndexValuePair<T> t{i + threadIdx.x, vals[i + threadIdx.x]};
        val = reduce_max_pair(val, t);
      }
    }

    for(size_t i = 16;i > 0;i >>= 1) {
        IndexValuePair<T> t{__shfl_down_sync(0xffffffff, val.idx, i), __shfl_down_sync(0xffffffff, val.val, i)};
        val = reduce_max_pair(val, t);
    }

    int laneid = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;
    if(laneid == 0) {
        s_mem[warpid] = val;
    }
    __syncthreads();

    if(warpid == 0) {
        int warpnum = blockDim.x / 32;
        val = laneid < warpnum ? s_mem[laneid] : s_mem[0];
        for(size_t i = 16;i > 0;i >>= 1) {
            IndexValuePair<T> t{__shfl_down_sync(0xffffffff, val.idx, i), __shfl_down_sync(0xffffffff, val.val, i)};
            val = reduce_max_pair(val, t);
        }
        if(laneid == 0) {
          *max_idx = val.idx;
          *max_val = val.val;
        }
    }
}

	
} // anonymous namespace


namespace llaisys::ops::nvidia {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
			llaisysDataType_t type, size_t numel) {
	
	auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
  dim3 block(1024);
  dim3 grid(1);
  switch (type) {
    case LLAISYS_DTYPE_F32: {
      argmax_<float><<<grid, block, 32 * sizeof(IndexValuePair<float>), s>>>(
        reinterpret_cast<int64_t*>(max_idx), reinterpret_cast<float*>(max_val), reinterpret_cast<const float*>(vals), numel);
      break;
    }
    case LLAISYS_DTYPE_F16: {
      argmax_<__half><<<grid, block, 32 * sizeof(IndexValuePair<__half>), s>>>(
        reinterpret_cast<int64_t*>(max_idx), reinterpret_cast<__half*>(max_val), reinterpret_cast<const __half*>(vals), numel);
      break;
    }
    case LLAISYS_DTYPE_BF16: {
      argmax_<__nv_bfloat16><<<grid, block, 32 * sizeof(IndexValuePair<__nv_bfloat16>), s>>>(
        reinterpret_cast<int64_t*>(max_idx), reinterpret_cast<__nv_bfloat16*>(max_val), reinterpret_cast<const __nv_bfloat16*>(vals), numel);
      break;
    }
    default: {
      EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
  }
}

} // namespace llaisys::ops::nvidia