#include "llaisys.h"
#include "../../ops.hpp"
#include "../../../device/nvidia/utils.cuh"


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>



namespace {
#define WARP_SIZE 32
#define CEIL(a, b) (((a) + (b) - 1) / (b))

template<typename T>
__global__ void flash_attention_(T* out, const T* q, const T* k, const T* v, const int Br, const int Bc, const int Tr, const int Tc, const float scale, const size_t q_len,
                                const size_t kv_len, const size_t nhead, const size_t nkv_head, const size_t dim, float* l, float* m) {
    extern __shared__ float s_mem[];
    float* s_q = s_mem;
    float* s_k = s_q + Br * dim;
    float* s_v = s_k + Bc * dim;
    float* s_S = s_v + Br * dim;
    int head_id = blockIdx.x;
    int kvhead_id = head_id / (nhead / nkv_head);
    int tx = threadIdx.x;
    l += head_id * q_len;
    m += head_id * q_len;
    for(size_t tc = 0;tc < Tc;tc++) {
      for(size_t i = 0;i < dim;i++) {
        s_k[tx * dim + i] = Bc * tc + tx < kv_len ? static_cast<float>(k[(Bc * tc + tx) * nkv_head * dim + kvhead_id * dim + i]) : 0.f;
        s_v[tx * dim + i] = Bc * tc + tx < kv_len ? static_cast<float>(v[(Bc * tc + tx) * nkv_head * dim + kvhead_id * dim + i]) : 0.f;
        //printf("%f %f\n", s_k[tx * dim + i], k[(Bc * tc + tx) * nkv_head * dim + kvhead_id * dim + i]);
        //printf("%f %f\n", s_v[tx * dim + i], v[(Bc * tc + tx) * nkv_head * dim + kvhead_id * dim + i]);
      }
      __syncthreads();
      for(size_t tr = 0;tr < Tr;tr++) {
        for(size_t i = 0;i < dim;i++) {
          s_q[tx * dim + i] = Br * tr + tx < q_len ? static_cast<float>(q[(Br * tr + tx) * nhead * dim + head_id * dim + i]) : 0.f;
        }
        __syncthreads();
        float now_m = -__FLT_MAX__;
        for(size_t c = 0;c < Bc;c++) {
          float sum = 0.f;
          if(Bc * tc + c > tr * Br + tx - q_len + kv_len) {
            s_S[tx * Bc + c] = 0.f;
            continue;
          } 
          for(size_t i = 0;i < dim;i++) {
            sum += s_q[tx * dim + i] * s_k[c * dim + i]; 
          }
          sum *= scale;
          now_m = fmaxf(now_m, sum);
          s_S[tx * Bc + c] = sum;
        }
        float now_l = 0.f;
        for(size_t c = 0;c < Bc;c++) {
          if(Bc * tc + c > tr * Br + tx - q_len + kv_len) {
            break;
          }
          s_S[tx * Bc + c] = expf(s_S[tx * Bc + c] - now_m);
          now_l += s_S[tx * Bc + c];
        }
        float pre_m = tc == 0 ? -__FLT_MAX__ : m[tr * Br + tx];
        float pre_l = tc == 0 ? 0.f : l[tr * Br + tx];
        float new_m = fmaxf(pre_m, now_m);
        float new_l = pre_l * expf(pre_m - new_m) + now_l * expf(now_m - new_m);
        for(size_t d = 0;d < dim;d++) {
          float sum = 0.f;
          for(size_t c = 0;c < Bc;c++) {
            //printf("%f %f\n", s_S[tx * Bc + c], s_v[c * dim + d]);
            sum += s_S[tx * Bc + c] * s_v[c * dim + d];
          }
          //printf("%f\n", sum);
          if(tr * Br + tx >= q_len) break;
          int idx = (tr * Br + tx) * nhead * dim + head_id * dim + d;
          float tmp = tc == 0 ? 0.f : static_cast<float>(out[idx]);
          out[idx] = static_cast<T>((tmp * expf(pre_m - new_m) * pre_l + sum * expf(now_m - new_m)) / new_l);
          m[tr * Br + tx] = new_m;
          l[tr * Br + tx] = new_l;
        }
      }
      __syncthreads();
    }
}
}


namespace llaisys::ops::nvidia {

void flash_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
            const float scale, const size_t q_len, const size_t kv_len, const size_t nhead, const size_t nkv_head, const size_t dim,
            llaisysDataType_t type, std::byte *l, std::byte *m) {
    //return ;
    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    const int Bc = 16;
    const int Br = 16;
    const int Tr = CEIL(q_len, Br);
    const int Tc = CEIL(kv_len, Bc);
    dim3 block(Br);
    dim3 grid(nhead);
    size_t s_mem = (3 * Br * dim + Br * Bc) * sizeof(float);
    //printf("enter flash_attention\n");
    switch (type) {
		case LLAISYS_DTYPE_F32:
      //printf("enter cuda float\n");
			flash_attention_<float><<<grid, block, s_mem, s>>>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k),
                                                reinterpret_cast<const float*>(v), Br, Bc, Tr, Tc, scale, q_len, kv_len, nhead, nkv_head, dim, 
                                                reinterpret_cast<float*>(l), reinterpret_cast<float*>(m));
      //printf("cuda float execute successful\n");
			break;
		case LLAISYS_DTYPE_F16:
      //printf("enter cuda fp16\n");
			flash_attention_<__half><<<grid, block, s_mem, s>>>(reinterpret_cast<__half*>(out), reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(k),
                                                reinterpret_cast<const __half*>(v), Br, Bc, Tr, Tc, scale, q_len, kv_len, nhead, nkv_head, dim, 
                                                reinterpret_cast<float*>(l), reinterpret_cast<float*>(m));
			//printf("cuda fp16 execute successful\n");
      break;
		case LLAISYS_DTYPE_BF16:
			//printf("enter cuda bf16\n");
      flash_attention_<__nv_bfloat16><<<grid, block, s_mem, s>>>(reinterpret_cast<__nv_bfloat16*>(out), reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k),
                                                reinterpret_cast<const __nv_bfloat16*>(v), Br, Bc, Tr, Tc, scale, q_len, kv_len, nhead, nkv_head, dim, 
                                                reinterpret_cast<float*>(l), reinterpret_cast<float*>(m));
      //printf("cuda bf16 execute successful\n");
      break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}