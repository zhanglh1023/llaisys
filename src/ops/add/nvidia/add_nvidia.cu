#include "add_nvidia.cuh"
#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../device/nvidia/utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define C_FLOAT4(value) (reinterpret_cast<const float4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define C_HALF2(value) (reinterpret_cast<const half2*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define C_BFLOAT2(value) (reinterpret_cast<const __nv_bfloat162*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define C_LDST128BIT(value) (reinterpret_cast<const float4*>(&(value))[0])
#define LDST128BIT(value) (reinterpret_cast<float4*>(&(value))[0])
namespace {

__global__ void addFloat4_kernel(float* c, const float* a, const float* b, size_t n) {
    size_t idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if(idx >= n) return ;
    if(idx + 4 < n) {
        const float4 a_t = C_FLOAT4(a[idx]);
        const float4 b_t = C_FLOAT4(b[idx]);
        float4 c_t;
        c_t.x = a_t.x + b_t.x;
        c_t.y = a_t.y + b_t.y;
        c_t.z = a_t.z + b_t.z;
        c_t.w = a_t.w + b_t.w;
        FLOAT4(c[idx]) = c_t;
        return ;
    }
    #pragma unroll
    for(size_t i = 0;idx + i < n;i++)
        c[idx + i] = a[idx + i] + b[idx + i];
}

__global__ void addfp16X8_kernel(half* c, const half* a, const half* b, size_t n) {
    size_t idx = (blockDim.x * blockIdx.x + threadIdx.x) * 8;
    if(idx >= n) return ;
    half pack_a[8], pack_b[8], pack_c[8];
    if(idx + 8 < n) {
        LDST128BIT(pack_a[0]) = C_LDST128BIT(a[idx]);
        LDST128BIT(pack_b[0]) = C_LDST128BIT(b[idx]);
        #pragma unroll
        for(size_t i = 0;i < 8;i += 2) {
            HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
        }
        LDST128BIT(c[idx]) = LDST128BIT(pack_c[0]);
        return ;
    }
    #pragma unroll
    for(size_t i = 0;idx + i + 1 <= n;i += 2) {
        if(idx + i + 1 == n) {
            pack_c[i] = __hadd(a[idx + i], b[idx + i]);
        } else {
            HALF2(pack_c[i]) = __hadd2(C_HALF2(a[idx + i]), C_HALF2(b[idx + i]));
        }
    }
    #pragma unroll
    for(size_t i = 0;idx + i < n;i++) {
        c[idx + i] = pack_c[i];
    }
}


__global__ void addbf16X8_kernel(__nv_bfloat16* c, const __nv_bfloat16* a, const __nv_bfloat16* b, size_t n) {
    size_t idx = (blockDim.x * blockIdx.x + threadIdx.x) * 8;
    if(idx >= n) return ;
    __nv_bfloat16 pack_a[8], pack_b[8], pack_c[8];
    if(idx + 8 < n) {
        LDST128BIT(pack_a[0]) = C_LDST128BIT(a[idx]);
        LDST128BIT(pack_b[0]) = C_LDST128BIT(b[idx]);
        #pragma unroll
        for(size_t i = 0;i < 8;i += 2) {
            BFLOAT2(pack_c[i]) = __hadd2(BFLOAT2(pack_a[i]), BFLOAT2(pack_b[i]));
        }
        LDST128BIT(c[idx]) = LDST128BIT(pack_c[0]);
        return ;
    }
    #pragma unroll
    for(size_t i = 0;idx + i + 1 <= n;i += 2) {
        if(idx + i + 1 == n) {
            pack_c[i] = __hadd(a[idx + i], b[idx + i]);
        } else {
            BFLOAT2(pack_c[i]) = __hadd2(C_BFLOAT2(a[idx + i]), C_BFLOAT2(b[idx + i]));
        }
    }
    #pragma unroll
    for(size_t i = 0;idx + i < n;i++) {
        c[idx + i] = pack_c[i];
    }
}
} // anonymous namespace


namespace llaisys::ops::nvidia {

void add(std::byte* c, const std::byte* a, const std::byte* b,
                llaisysDataType_t type, size_t n) {

  auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());


  dim3 block(256), grid(safe_grid_size(n, block.x)); // ceil division

  switch (type) {
    case LLAISYS_DTYPE_F32: {
      addFloat4_kernel<<<CEIL(CEIL(n, 4), 256), 256, 0, s>>>(
        reinterpret_cast<float*>(c),
        reinterpret_cast<const float*>(a),
        reinterpret_cast<const float*>(b),
        n);
      break;
    }
    case LLAISYS_DTYPE_F16: {
      addfp16X8_kernel<<<CEIL(CEIL(n, 8), 256), 256, 0, s>>>(
        reinterpret_cast<__half*>(c),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(b),
        n);
      break;
    }
    case LLAISYS_DTYPE_BF16: {
      addbf16X8_kernel<<<CEIL(CEIL(n, 8), 256), 256, 0, s>>>(
        reinterpret_cast<__nv_bfloat16*>(c),
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        n);
      break;
    }
    default:
      EXCEPTION_UNSUPPORTED_DATATYPE(type); 
  }
}


} // namespace llaisys::ops::nvidia