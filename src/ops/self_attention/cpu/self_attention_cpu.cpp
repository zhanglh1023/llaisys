#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, const float scale,
                    const std::vector<size_t>& attn_val_shape, const std::vector<ptrdiff_t>& attn_val_strides, 
                    const std::vector<size_t>& q_shape, const std::vector<ptrdiff_t>& q_strides,
                    const std::vector<size_t>& k_shape, const std::vector<ptrdiff_t>& k_strides,
                    const std::vector<size_t>& v_shape, const std::vector<ptrdiff_t>& v_strides) {
    size_t seq_len = q_shape[0];
    size_t nqhead = q_shape[1];
    size_t hidden_size = q_shape[2];
    size_t total_len = k_shape[0];
    size_t nkvhead = k_shape[1];
    for(size_t seq_id = 0;seq_id < seq_len;seq_id++) {
        for(size_t qhead_id = 0;qhead_id < nqhead;qhead_id++) {
            std::vector<float> A(total_len, 0);
            size_t kvhead_id = qhead_id * nkvhead / nqhead;
            size_t causal_len = total_len - seq_len + seq_id + 1;
            float mx = -1e9f;
            for(size_t total_id = 0;total_id < causal_len;total_id++) {
                for(size_t i = 0;i < hidden_size;i++) {
                    size_t q_idx = seq_id * q_strides[0] + qhead_id * q_strides[1] + i * q_strides[2];
                    size_t k_idx = total_id * k_strides[0] + kvhead_id * k_strides[1] + i * k_strides[2];
                    A[total_id] += llaisys::utils::cast<float>(q[q_idx]) * llaisys::utils::cast<float>(k[k_idx]);
                }
                A[total_id] *= scale;
                mx = std::max(mx, A[total_id]);
            }
            float sum = 0;
            for(size_t i = 0;i < causal_len;i++) {
                sum += expf(A[i] - mx);
            }
            for(size_t i = 0;i < causal_len;i++) {
                A[i] = expf(A[i] - mx) / sum;
            }
            size_t dv = v_shape[2];
            for(size_t i = 0;i < dv;i++) {
                float x = 0;
                for(size_t j = 0;j < causal_len;j++) {
                    x += A[j] * llaisys::utils::cast<float>(v[j * v_strides[0] + kvhead_id * v_strides[1] + i * v_strides[2]]);
                }
                attn_val[seq_id * attn_val_strides[0] + qhead_id * attn_val_strides[1] + i * attn_val_strides[2]]
                    = llaisys::utils::cast<T>(x);
            }
        }
    }

}


namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, const float scale,
                    llaisysDataType_t type, const std::vector<size_t>& attn_val_shape, const std::vector<ptrdiff_t>& attn_val_strides, 
                    const std::vector<size_t>& q_shape, const std::vector<ptrdiff_t>& q_strides,
                    const std::vector<size_t>& k_shape, const std::vector<ptrdiff_t>& k_strides,
                    const std::vector<size_t>& v_shape, const std::vector<ptrdiff_t>& v_strides) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float*>(attn_val), reinterpret_cast<const float*>(q), 
                reinterpret_cast<const float*> (k), reinterpret_cast<const float*> (v), scale, attn_val_shape,
                attn_val_strides, q_shape, q_strides, k_shape, k_strides, v_shape, v_strides);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t*>(attn_val), reinterpret_cast<const llaisys::bf16_t*>(q), 
                reinterpret_cast<const llaisys::bf16_t*>(k), reinterpret_cast<const llaisys::bf16_t*>(v), scale, attn_val_shape,
                attn_val_strides, q_shape, q_strides, k_shape, k_strides, v_shape, v_strides);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t*>(attn_val), reinterpret_cast<const llaisys::fp16_t*>(q), 
                reinterpret_cast<const llaisys::fp16_t*>(k), reinterpret_cast<const llaisys::fp16_t*>(v), scale, attn_val_shape,
                attn_val_strides, q_shape, q_strides, k_shape, k_strides, v_shape, v_strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

}
}