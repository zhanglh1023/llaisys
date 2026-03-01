#include "llaisys.h"
#include "../../ops.hpp"
#include "../../../device/nvidia/utils.cuh"


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>



namespace {
#define CEIL(N, M) (((N) + (M) - 1) / (M))
#define LDST128BITS(v) (reinterpret_cast<float4*>(&(v))[0])
#define WARP_SZIE 32

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T value) {
  #pragma unroll
  for(size_t i = WARP_SZIE >> 1;i > 0;i >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, i);
  }
  return value;
}
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T value) {
  #pragma unroll
  for(size_t i = WARP_SZIE >> 1;i > 0;i >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, i));
  }
  return value;
}

template<typename T, const int Br = 16, const int Bc = 32, 
        const int TM = 1, const int TN = 2, const int BD = 8, const int padding = 0>
__global__ void flash_attn_kernel(const T *Q, const T *K, const T *V, T *O, 
                          const int q_len, const int kv_len, const int kv_heads, const int dim, const bool is_causal, const float scale) {
  const int BM = Br * TM;
  const int BN = Bc * TN;
  const int bx = blockIdx.x;
  const int block_size = blockDim.x;
  const int head_id = blockIdx.y;
  const int kv_head_id = head_id / (gridDim.y / kv_heads);
  const int batch_id = blockIdx.z;
  const int tid = threadIdx.x;
  const int tx = tid % Bc;
  const int ty = tid / Bc;
  const int laneid = tid % WARP_SZIE;
  const int q_stride = gridDim.y * dim;
  const int kv_stride = kv_heads * dim;
  const T* q = Q + batch_id * q_len * q_stride + head_id * dim + bx * BM * q_stride;
  const T* k = K + batch_id * kv_len * kv_stride + kv_head_id * dim;
  const T* v = V + batch_id * kv_len * kv_stride + kv_head_id * dim;
  T* o = O + batch_id * q_len * q_stride + head_id * dim + bx * BM * q_stride;
  
  //kv_len - q_len + q_acc_len + ty >= kv_acc_len + tx
  const int Tc = CEIL((is_causal ? min(kv_len - q_len + (bx + 1) * BM, kv_len) : kv_len), BN);

  extern __shared__ char smem[];
  
  // shared_mem shape: for bcf 
  // q o: Br * dim
  // k v: dim * Bc
  float *s_q = (float*)smem;
  float *s_kv = s_q + BM * dim;
  float *s_o = s_kv + (BN + padding) * BD * 2;
  float *s_m = s_o + BM * dim;
  float *s_l = s_m + BM;
  
  #pragma unroll
  for(size_t i = tid;i < BM;i += block_size) {
    s_m[i] = -__FLT_MAX__;
    s_l[i] = 0.f;
  }
  int q_acc_len = bx * BM;
  int kv_acc_len = 0;

  #pragma unroll
  for(size_t i = tid;i < BM * dim;i += block_size) {
    int x = i % dim;
    int y = i / dim;
    s_q[y * dim + x] = ((q_acc_len + y) < q_len) ? static_cast<float>(q[y * q_stride + x]) : float(0);
    s_o[i] = 0.f;
  }
  #pragma unroll
  for(size_t c = 0;c < Tc;++c) {
    
    // sum[0]: ty tx 、 sum[1]: ty tx + Bc
    float sum[TM][TN] = {0.f};
    #pragma unroll
    for(size_t d = 0;d < dim;d += BD) {
        int idx = (d / BD) % 2;
        #pragma unroll
        for(size_t i = tid;i < BD * BN;i+=block_size) {
            int s_x = i % BD;
            int y = i / BD;
            // s_x * (BN + paddingk) + y
            s_kv[idx * (BN + padding) * BD + s_x * (BN + padding) + y] = (kv_acc_len + y < kv_len && s_x + d < dim) ? static_cast<float>(k[y * kv_stride + s_x + d]) : float(0);
        }
        __syncthreads();
        #pragma unroll
        for(size_t i = 0;i < BD;++i) {
            float q_reg[TM] = {0.f};
            float k_reg[TN] = {0.f};
            #pragma unroll
            for(size_t j = 0;j < TM;j++)
                q_reg[j] = (d + i < dim) ? s_q[(ty * TM + j) * dim + d + i] : 0.f;//(ty * TM + j) * dim + d + i
            #pragma unroll
            for(size_t j = 0;j < TN;j+=4)
                LDST128BITS(k_reg[j]) = LDST128BITS(s_kv[idx * (BN + padding) * BD + i * (BN + padding) + tx * 4 + j * Bc]);//i * (BN + paddingk) + tx * TN + j
            
            #pragma unroll
            for(size_t j = 0;j < TM;j++) {
                #pragma unroll
                for(size_t k = 0;k < TN;k++) {
                    sum[j][k] += q_reg[j] * k_reg[k];
                }
            }
        }
       // __syncthreads();
    }
    
    float l_pre[TM];//tile每行上一个Bc的和
    float l[TM];//加上当前Bc整体tile行的和
    float p_now[TM][TN];
    float exp_mprem[TM];
    float exp_mnowm[TM];
    #pragma unroll
    for(size_t i = 0;i < TM;i++) {
        float m_sum; //带casual掩码的sum
        float m_now = -__FLT_MAX__;//tile每行的最大值
        float l_now = 0.f;//tile每行的和
        float m_pre = s_m[ty * TM + i];//tile每行上一个Bc的最大值
        l_pre[i] = s_l[ty * TM + i];
        float m = m_pre;//加上当前Bc整体tile行的最大值
        l[i] = l_pre[i];
        #pragma unroll
        for(size_t j = 0;j < TN;++j) {
            sum[i][j] *= scale;
            m_sum = (((q_acc_len + ty * TM + i < q_len) && (kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4 < kv_len)) && (!is_causal || kv_len - q_len + q_acc_len + ty * TM + i >= kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4)) ? sum[i][j] : -__FLT_MAX__;
            if(((q_acc_len + ty * TM + i < q_len) && (kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4 < kv_len)) && (!is_causal || kv_len - q_len + q_acc_len + ty * TM + i >= kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4))
                m_now = fmaxf(m_now, m_sum);
        }
        m_now = warp_reduce_max<float>(m_now);
        for(size_t j = 0;j < TN;++j) {
            sum[i][j] = (((q_acc_len + ty * TM + i < q_len) && (kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4 < kv_len)) && (!is_causal || kv_len - q_len + q_acc_len + ty * TM + i >= kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4)) ? expf(sum[i][j]-m_now) : 0.f;
            if(((q_acc_len + ty * TM + i < q_len) && (kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4 < kv_len)) && (!is_causal || kv_len - q_len + q_acc_len + ty * TM + i >= kv_acc_len + tx * 4 + j % 4 + j / 4 * Bc * 4))
                l_now += sum[i][j];
        }
        l_now = warp_reduce_sum<float>(l_now);
        #pragma unroll
        for(size_t j = 0;j < TN;j++)
            p_now[i][j] = sum[i][j];
        m = fmaxf(m, m_now);
        exp_mprem[i] = expf(m_pre - m);
        exp_mnowm[i] = expf(m_now - m);
        l[i] = l_pre[i] * exp_mprem[i] + l_now * exp_mnowm[i];
        s_m[ty * TM + i] = m;
        s_l[ty * TM + i] = l[i];
    }
    
    #pragma unroll
    for(size_t d = 0;d < dim;d += BD) {
        int idx = (d / BD) % 2;
        #pragma unroll
        for(size_t i = tid;i < BD * BN;i += block_size) {
            int s_x = i % BD;
            int y = i / BD;
            //s_x * (BN + paddingv) + y
            s_kv[idx * (BN + padding) * BD + s_x * (BN + padding) + y] = ((kv_acc_len + y) < kv_len && s_x + d < dim) ? static_cast<float>(v[y * kv_stride + s_x + d]) : float(0);
        }
        __syncthreads();
        #pragma unroll
        for(size_t j = 0;j < BD;++j) {
            float value[TM] = {0.f};
            #pragma unroll
            for(size_t i = 0;i < TN;i+=4) {
                float4 tmp = LDST128BITS(s_kv[idx * (BN + padding) * BD + j * (BN + padding) + tx * 4 + i * Bc]);//j * (BN + paddingv) + tx * TN + i
                #pragma unroll
                for(size_t k = 0;k < TM;k++) {
                    value[k] += p_now[k][i] * tmp.x + p_now[k][i + 1] * tmp.y + p_now[k][i + 2] * tmp.z + p_now[k][i + 3] * tmp.w;
                }
            }
            #pragma unroll
            for(size_t i = 0;i < TM;i++) {
                value[i] = warp_reduce_sum<float>(value[i]);
                if(laneid == 0 && j + d < dim)
                    s_o[(ty * TM + i) * dim + j + d] = (q_acc_len + ty * TM + i < q_len) ? (s_o[(ty * TM + i) * dim + j + d] * exp_mprem[i] * l_pre[i] + value[i] * exp_mnowm[i]) / l[i] : 0.f;   
            }
        }
    }
    k += BN * kv_stride;
    v += BN * kv_stride;
    kv_acc_len += BN;
  }
  __syncthreads();
  #pragma unroll
  for(size_t i = tid;i < BM * dim;i += block_size) {
    int x = i % dim;
    int y = i / dim;
    if(q_acc_len + y < q_len) {
        o[y * q_stride + x] = static_cast<T>(s_o[y * dim + x]);
    }
  }
}

}

namespace llaisys::ops::nvidia {

void flash_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
            const float scale, const size_t q_len, const size_t kv_len, const size_t nhead, const size_t nkv_head, const size_t dim,
            llaisysDataType_t type) {
    //return ;
    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    
    constexpr int Br = 16;
    constexpr int Bc = 32;
    constexpr int TM = 1;
    constexpr int TN = 8;
    constexpr int BD = 8;
    constexpr int padding = 4;
    dim3 block(Br * Bc);
    dim3 grid(CEIL(q_len, Br * TM), nhead, 1);
    int s_mem = ((Br * TM) * dim * 2 + (Bc * TN + padding) * BD * 2 + Br * TM * 2) * sizeof(float);
    
    switch (type) {
		case LLAISYS_DTYPE_F32:
      //printf("enter cuda float\n");
			flash_attn_kernel<float, Br, Bc, TM, TN, BD, padding><<<grid, block, s_mem, s>>>(reinterpret_cast<const float*>((q)), reinterpret_cast<const float*>((k)), reinterpret_cast<const float*>((v)),
                                                reinterpret_cast<float*>(out), q_len, kv_len, nkv_head, dim, true, scale);
      //printf("cuda float execute successful\n");
			break;
		case LLAISYS_DTYPE_F16:
      //printf("enter cuda fp16\n");
			flash_attn_kernel<__half, Br, Bc, TM, TN, BD, padding><<<grid, block, s_mem, s>>>(reinterpret_cast<const __half*>((q)), reinterpret_cast<const __half*>((k)), reinterpret_cast<const __half*>((v)),
                                                reinterpret_cast<__half*>(out), q_len, kv_len, nkv_head, dim, true, scale);
			//printf("cuda fp16 execute successful\n");
      break;
		case LLAISYS_DTYPE_BF16:
			//printf("enter cuda bf16\n");
      flash_attn_kernel<__nv_bfloat16, Br, Bc, TM, TN, BD, padding><<<grid, block, s_mem, s>>>(reinterpret_cast<const __nv_bfloat16*>((q)), reinterpret_cast<const __nv_bfloat16*>((k)), reinterpret_cast<const __nv_bfloat16*>((v)),
                                                reinterpret_cast<__nv_bfloat16*>(out), q_len, kv_len, nkv_head, dim, true, scale);
      //printf("cuda bf16 execute successful\n");
      break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}