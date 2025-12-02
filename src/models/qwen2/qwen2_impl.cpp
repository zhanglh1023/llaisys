#include "qwen2_impl.hpp"

#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include "../../ops/ops.hpp"

#include <chrono>
#include <functional>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>
namespace llaisys::models::qwen2 {

static void alloc_weight_arrays(LlaisysQwen2Weights &weights, size_t nlayer) {
    weights.attn_norm_w = new llaisysTensor_t[nlayer];
    weights.attn_q_w = new llaisysTensor_t[nlayer];
    weights.attn_q_b = new llaisysTensor_t[nlayer];
    weights.attn_k_w = new llaisysTensor_t[nlayer];
    weights.attn_k_b = new llaisysTensor_t[nlayer];
    weights.attn_v_w = new llaisysTensor_t[nlayer];
    weights.attn_v_b = new llaisysTensor_t[nlayer];
    weights.attn_o_w = new llaisysTensor_t[nlayer];
    weights.mlp_norm_w = new llaisysTensor_t[nlayer];
    weights.mlp_gate_w = new llaisysTensor_t[nlayer];
    weights.mlp_up_w = new llaisysTensor_t[nlayer];
    weights.mlp_down_w = new llaisysTensor_t[nlayer];
    for (size_t i = 0; i < nlayer; i++) {
        weights.attn_norm_w[i] = nullptr;
        weights.attn_q_w[i] = nullptr;
        weights.attn_q_b[i] = nullptr;
        weights.attn_k_w[i] = nullptr;
        weights.attn_k_b[i] = nullptr;
        weights.attn_v_w[i] = nullptr;
        weights.attn_v_b[i] = nullptr;
        weights.attn_o_w[i] = nullptr;
        weights.mlp_norm_w[i] = nullptr;
        weights.mlp_gate_w[i] = nullptr;
        weights.mlp_up_w[i] = nullptr;
        weights.mlp_down_w[i] = nullptr;
    }
    weights.in_embed = weights.out_embed = weights.out_norm_w = nullptr;
}
static void free_weight_arrays_keep_tensors(LlaisysQwen2Weights &weights) {
    if (weights.attn_norm_w) {
        delete[] weights.attn_norm_w;
        weights.attn_norm_w = nullptr;
    }
    if (weights.attn_q_w) {
        delete[] weights.attn_q_w;
        weights.attn_q_w = nullptr;
    }
    if (weights.attn_q_b) {
        delete[] weights.attn_q_b;
        weights.attn_q_b = nullptr;
    }
    if (weights.attn_k_w) {
        delete[] weights.attn_k_w;
        weights.attn_k_w = nullptr;
    }
    if (weights.attn_k_b) {
        delete[] weights.attn_k_b;
        weights.attn_k_b = nullptr;
    }
    if (weights.attn_v_w) {
        delete[] weights.attn_v_w;
        weights.attn_v_w = nullptr;
    }
    if (weights.attn_v_b) {
        delete[] weights.attn_v_b;
        weights.attn_v_b = nullptr;
    }
    if (weights.attn_o_w) {
        delete[] weights.attn_o_w;
        weights.attn_o_w = nullptr;
    }
    if (weights.mlp_norm_w) {
        delete[] weights.mlp_norm_w;
        weights.mlp_norm_w = nullptr;
    }
    if (weights.mlp_gate_w) {
        delete[] weights.mlp_gate_w;
        weights.mlp_gate_w = nullptr;
    }
    if (weights.mlp_up_w) {
        delete[] weights.mlp_up_w;
        weights.mlp_up_w = nullptr;
    }
    if (weights.mlp_down_w) {
        delete[] weights.mlp_down_w;
        weights.mlp_down_w = nullptr;
    }
    weights.in_embed = weights.out_embed = weights.out_norm_w = nullptr;
}
static void free_all_weights_and_tensors(LlaisysQwen2Weights &weights, size_t nlayer) {
    std::function<void(llaisysTensor_t)> destroy =
        [](llaisysTensor_t t) {
            if (t) {
                tensorDestroy(t);
            }
        };
    destroy(weights.in_embed);
    destroy(weights.out_embed);
    destroy(weights.out_norm_w);
    if (weights.attn_norm_w) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.attn_norm_w[i]);
        }
    }
    if (weights.attn_q_w) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.attn_q_w[i]);
        }
    }
    if (weights.attn_q_b) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.attn_q_b[i]);
        }
    }
    if (weights.attn_k_w) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.attn_k_w[i]);
        }
    }
    if (weights.attn_k_b) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.attn_k_b[i]);
        }
    }
    if (weights.attn_v_w) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.attn_v_w[i]);
        }
    }
    if (weights.attn_v_b) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.attn_v_b[i]);
        }
    }
    if (weights.attn_o_w) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.attn_o_w[i]);
        }
    }
    if (weights.mlp_norm_w) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.mlp_norm_w[i]);
        }
    }
    if (weights.mlp_gate_w) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.mlp_gate_w[i]);
        }
    }
    if (weights.mlp_up_w) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.mlp_up_w[i]);
        }
    }
    if (weights.mlp_down_w) {
        for (size_t i = 0; i < nlayer; i++) {
            destroy(weights.mlp_down_w[i]);
        }
    }
    free_weight_arrays_keep_tensors(weights);
}
Qwen2Impl::Qwen2Impl(const LlaisysQwen2Meta &meta_,
                     const llaisysDeviceType_t &device_,
                     const int *device_ids_,
                     const int ndevice_) : meta(meta_), device(device_) {
    if (device_ids_ && ndevice_ > 0) {
        device_ids.assign(device_ids_, device_ids_ + ndevice_);
    }
    alloc_weight_arrays(weights, meta.nlayer);
}
Qwen2Impl::~Qwen2Impl() {
    free_all_weights_and_tensors(weights, meta.nlayer);
}

static tensor_t llaisysTensorTotensor(llaisysTensor_t x) {
    struct LlaisysTensor* t = reinterpret_cast<LlaisysTensor*>(x);
    return t->tensor;
}


template<typename T>
static T read_scalar_from_tensor(const tensor_t& t) {
    T v{};
    if (t->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(&v, t->data(), sizeof(T));
    } else {
        core::context().runtime().api()->memcpy_sync(&v, t->data(), sizeof(T), LLAISYS_MEMCPY_D2H);
    }
    return v;
}
static size_t safe_dim(tensor_t t, size_t idx) {
    if(t == nullptr) return 0;
    const std::vector<size_t>& shape = t->shape();
    if(idx < 0 || idx >= shape.size()) return 0;
    return shape[idx];
}
int64_t Qwen2Impl::forward(const int64_t *token_ids, size_t ntoken, size_t pos_base) {
    std::cerr << "[Forward] start, ntoken=" << ntoken << " pos_base=" << pos_base << std::endl;
    if (!weights.in_embed || !weights.out_embed || !weights.out_norm_w) {
        std::cerr << "[Forward] ERROR: missing essential weights" << std::endl;
        return (ntoken ? token_ids[ntoken-1] : (meta.end_token >= 0 ? meta.end_token : 0));
    }
    tensor_t in_embed = llaisysTensorTotensor(weights.in_embed);
    tensor_t out_embed = llaisysTensorTotensor(weights.out_embed);
    tensor_t out_norm = llaisysTensorTotensor(weights.out_norm_w);
    llaisysDataType_t dtype = meta.dtype;
    
    size_t total_len = safe_dim(in_embed, 0);
    size_t hidden_size = safe_dim(in_embed, 1);
    if(!total_len || !hidden_size) {
        std::cerr << "[Forward] ERROR: invalid in_embed dims" << std::endl;
        return (ntoken ? token_ids[ntoken-1] : (meta.end_token >= 0 ? meta.end_token : 0));
    }
    
    for (size_t i = 0;i < ntoken;i++) {
        if(token_ids[i] < 0 || token_ids[i] >= (int64_t)total_len) {
            std::cerr << "[Forward] ERROR: token_ids[" << i << "]=" << token_ids[i]
                      << " out of range vocab=" << total_len << std::endl;
            return (meta.end_token >= 0 ? meta.end_token : 0);
        }
    }
    
    llaisysDeviceType_t dev_type = in_embed->deviceType();
    int dev_id = in_embed->deviceId();

    size_t seqlen = ntoken;
    //embedding层
    tensor_t index = Tensor::create({seqlen}, LLAISYS_DTYPE_I64, dev_type, dev_id);
    index->load(token_ids);
    tensor_t x = Tensor::create({seqlen, hidden_size}, dtype, dev_type, dev_id);
    ops::embedding(x, index, in_embed);
    //std::cerr<<"index shape :" << safe_dim(index, 0)<<" "<<safe_dim(index, 1)<<"\n";
    //std::cerr<<"in_embed shape :" << safe_dim(in_embed, 0)<<" "<<safe_dim(in_embed, 1)<<"\n";
    tensor_t pos_ids = Tensor::create({seqlen}, LLAISYS_DTYPE_I64, dev_type, dev_id);
    {
        std::vector<int64_t> p(seqlen);
        for(size_t i = 0;i < seqlen;i++) p[i] = static_cast<int64_t>(pos_base + i);
        pos_ids->load(p.data());
    }
    
    //transformer block
    for(size_t layer_id = 0;layer_id < meta.nlayer;layer_id++) {
        tensor_t attn_norm = weights.attn_norm_w ? llaisysTensorTotensor(weights.attn_norm_w[layer_id]) : nullptr;
        tensor_t q_w = weights.attn_q_w ? llaisysTensorTotensor(weights.attn_q_w[layer_id]) : nullptr;
        tensor_t q_b = weights.attn_q_b ? llaisysTensorTotensor(weights.attn_q_b[layer_id]) : nullptr;
        tensor_t k_w = weights.attn_k_w ? llaisysTensorTotensor(weights.attn_k_w[layer_id]) : nullptr;
        tensor_t k_b = weights.attn_k_b ? llaisysTensorTotensor(weights.attn_k_b[layer_id]) : nullptr;
        tensor_t v_w = weights.attn_v_w ? llaisysTensorTotensor(weights.attn_v_w[layer_id]) : nullptr;
        tensor_t v_b = weights.attn_v_b ? llaisysTensorTotensor(weights.attn_v_b[layer_id]) : nullptr;
        tensor_t o_w = weights.attn_o_w ? llaisysTensorTotensor(weights.attn_o_w[layer_id]) : nullptr;
        tensor_t mlp_norm = weights.mlp_norm_w ? llaisysTensorTotensor(weights.mlp_norm_w[layer_id]) : nullptr;
        tensor_t gate_w = weights.mlp_gate_w ? llaisysTensorTotensor(weights.mlp_gate_w[layer_id]) : nullptr;
        tensor_t up_w = weights.mlp_up_w ? llaisysTensorTotensor(weights.mlp_up_w[layer_id]) : nullptr;
        tensor_t down_w = weights.mlp_down_w ? llaisysTensorTotensor(weights.mlp_down_w[layer_id]) : nullptr;
        if (!attn_norm || !q_w || !k_w || !v_w || !o_w || !mlp_norm || !gate_w || !up_w || !down_w) {
            std::cerr << "[Forward] missing weight(s) in layer " << layer_id << ", stop" << std::endl;
            break;
        }
        //attn_rms_norm
        tensor_t x_norm = Tensor::create({seqlen, hidden_size}, dtype, dev_type, dev_id);
        ops::rms_norm(x_norm, x, attn_norm, meta.epsilon);

        // Q/K/V
        size_t Qd = safe_dim(q_w, 0);
        size_t Kd = safe_dim(k_w, 0);
        size_t Vd = safe_dim(v_w, 0);
        if (Qd != meta.nh * meta.dh || Kd != meta.nkvh * meta.dh || Vd != meta.nkvh * meta.dh) {
            std::cerr << "[Forward] layer " << layer_id << " Q/K/V shape mismatch"
                      << " got (" << Qd << "," << Kd << "," << Vd << ")" << std::endl;
            break;
        }
        //if(layer_id == 0) {
            //std::cerr<<"q shape :" << safe_dim(q_w, 0)<<" "<<safe_dim(q_w, 1)<<"\n";
            //std::cerr<<"k shape :" << safe_dim(k_w, 0)<<" "<<safe_dim(k_w, 1)<<"\n";
            //std::cerr<<"v shape :" << safe_dim(v_w, 0)<<" "<<safe_dim(v_w, 1)<<"\n";
        //}
        tensor_t q2d = Tensor::create({seqlen, Qd}, dtype, dev_type, dev_id);
        tensor_t k2d = Tensor::create({seqlen, Kd}, dtype, dev_type, dev_id);
        tensor_t v2d = Tensor::create({seqlen, Vd}, dtype, dev_type, dev_id);
        ops::linear(q2d, x_norm, q_w, q_b);
        ops::linear(k2d, x_norm, k_w, k_b);
        ops::linear(v2d, x_norm, v_w, v_b);
        tensor_t q = q2d->view({seqlen, meta.nh, meta.dh});
        tensor_t k = k2d->view({seqlen, meta.nkvh, meta.dh});
        tensor_t v = v2d->view({seqlen, meta.nkvh, meta.dh});

        //rope
        ops::rope(q, q, pos_ids, meta.theta);
        ops::rope(k, k, pos_ids, meta.theta);

        tensor_t all_k = k;
        tensor_t all_v = v;
        if(use_kvcache) {
            append_kv(layer_id, k, v, pos_base);
            //std::cerr<<"append_kv\n";
            all_k = kcache_view(layer_id, pos_base + safe_dim(k, 0));
            all_v = vcache_view(layer_id, pos_base + safe_dim(v, 0));
        }
        //if(layer_id == 0) {
            //std::cerr<<"k_all shape :" << safe_dim(all_k, 0)<<" "<<safe_dim(all_k, 1)<<" "<<safe_dim(all_k, 2);
            //std::cerr<<"v_all shape :" << safe_dim(all_v, 0)<<" "<<safe_dim(all_v, 1)<<" "<<safe_dim(all_v, 2);
        //}
        //self_attention
        tensor_t attn3d = Tensor::create({seqlen, meta.nh, meta.dh}, dtype, dev_type, dev_id);

        float scale = 1.0 / std::sqrt((float)meta.dh);
        ops::self_attention(attn3d, q, all_k, all_v, scale);
        tensor_t attn = attn3d->view({seqlen, hidden_size});
        tensor_t o = Tensor::create({seqlen, hidden_size}, dtype, dev_type, dev_id);
        ops::linear(o, attn, o_w, nullptr);
        //if(layer_id == 0) {
            //std::cerr<<"o shape :" << safe_dim(o_w, 0)<<" "<<safe_dim(o_w, 1)<<"\n";
        //}
        tensor_t x_res1 = Tensor::create({seqlen, hidden_size}, dtype, dev_type, dev_id);

        ops::add(x_res1, x, o);

        // MLP
        tensor_t y_norm = Tensor::create({seqlen, hidden_size}, dtype, dev_type, dev_id);
        ops::rms_norm(y_norm, x_res1, mlp_norm, meta.epsilon);

        size_t intermediate_size = safe_dim(gate_w, 0);
        if(intermediate_size != meta.di) {
            std::cerr << "[Forward] layer " << layer_id << " MLP dim mismatch, expect "
                      << meta.di << " got " << intermediate_size << std::endl;
            break;
        }

        tensor_t gate = Tensor::create({seqlen, intermediate_size}, dtype, dev_type, dev_id);
        tensor_t up = Tensor::create({seqlen, intermediate_size}, dtype, dev_type, dev_id);
        ops::linear(gate, y_norm, gate_w, nullptr);
        ops::linear(up, y_norm, up_w, nullptr);
        tensor_t act = Tensor::create({seqlen, intermediate_size}, dtype, dev_type, dev_id);
        ops::swiglu(act, gate, up);
        tensor_t down = Tensor::create({seqlen, hidden_size}, dtype, dev_type, dev_id);
        ops::linear(down, act, down_w, nullptr);
        tensor_t x_res2 = Tensor::create({seqlen, hidden_size}, dtype, dev_type, dev_id);
        ops::add(x_res2, down, x_res1);
        x = x_res2;
    }
    
    tensor_t x_last = Tensor::create({seqlen, hidden_size}, dtype, dev_type, dev_id);
    ops::rms_norm(x_last, x, out_norm, meta.epsilon);
    tensor_t last_seq = x_last->slice(0, seqlen - 1, seqlen);
    size_t voc = safe_dim(out_embed, 0);
    tensor_t logits = Tensor::create({1, voc}, dtype, dev_type, dev_id);
    ops::linear(logits, last_seq, out_embed, nullptr);
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, dev_type, dev_id);
    tensor_t max_val = Tensor::create({1}, dtype, dev_type, dev_id);
    ops::argmax(max_idx, max_val, logits);
    
    //last_logits_ = logits;
    int64_t next_token_id = read_scalar_from_tensor<int64_t>(max_idx);
    std::cerr << "[Forward] done, next_token_id=" << next_token_id << std::endl;
    return next_token_id;
}
int64_t Qwen2Impl::prifill(const int64_t *token_ids, size_t ntoken) {
    auto start_time = std::chrono::high_resolution_clock::now();

    ctx_tokens_.assign(token_ids, token_ids + ntoken);
    reset_cache();

    int64_t next_token_id = forward(token_ids, ntoken, 0);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "[Prefill] Tokens: " << ntoken << ", Time: " << duration.count() / 1000 << " ms " << std::endl;
    std::cout << "next_token_id: " << next_token_id << std::endl;
    return next_token_id;
}

int64_t Qwen2Impl::decode_one(const int64_t token_id) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int64_t next_id = forward(&token_id, 1, ctx_tokens_.size());

    ctx_tokens_.push_back(token_id);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "[Decode] Tokens: " << ctx_tokens_.size() << ", Time: " << duration.count() / 1000 << " ms " << std::endl;
    std::cout << "next_token_id: " << next_id << std::endl;

    return next_id;
}

void Qwen2Impl::reset_cache() {
    k_cache.assign(meta.nlayer, nullptr);
    v_cache.assign(meta.nlayer, nullptr);
    cache_len = 0;
}

void Qwen2Impl::ensure_kvcache_capacity(size_t layer_id, size_t need_ntoken, llaisysDataType_t dtype, 
                            llaisysDeviceType_t devtype, int devid) {

    auto grow = [this, need_ntoken, dtype, devtype, devid](tensor_t& cache) {
        size_t old_capacity = safe_dim(cache, 0);
        size_t new_capacity = need_ntoken;

        if(new_capacity <= old_capacity) return ;

        old_capacity = old_capacity ? old_capacity : 1;
        while(old_capacity < new_capacity) old_capacity *= 2;
        
        tensor_t new_cache = Tensor::create({new_capacity, this->meta.nkvh, this->meta.dh}, dtype,
                                        devtype, devid);
        if(cache != nullptr) {
            size_t size = cache->numel() * cache->elementSize();
            auto api = core::context().runtime().api();
            api->memcpy_sync(new_cache->data(), cache->data(), size, 
                            (cache->deviceType() == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D);
        }
        cache = new_cache;
    };

    grow(k_cache[layer_id]);
    grow(v_cache[layer_id]);
}
void Qwen2Impl::append_kv(size_t layer_id, const tensor_t& k, const tensor_t& v, size_t pos) {
    
    size_t ntoken = safe_dim(k, 0);
    size_t nkvh = meta.nkvh;
    size_t dh = meta.dh;
    size_t row_size = dh * nkvh * k->elementSize();
    ensure_kvcache_capacity(layer_id, ntoken + pos, k->dtype(), k->deviceType(), k->deviceId());
    //std::cerr<<"ensure_kvcache\n";
    llaisysDeviceType_t src_device = k->deviceType();
    auto api = core::context().runtime().api();
    for(size_t i = 0;i < ntoken;i++) {
        auto dst = k_cache[layer_id]->data() + (pos + i) * row_size;
        auto src = k->data() + i * row_size;
        api->memcpy_sync(dst, src, row_size, ((src_device == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D));
    }
    for(size_t i = 0;i < ntoken;i++) {
        auto dst = v_cache[layer_id]->data() + (pos + i) * row_size;
        auto src = v->data() + i * row_size;
        api->memcpy_sync(dst, src, row_size, ((src_device == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D));
    }

    if(cache_len < ntoken + pos) cache_len  = ntoken + pos;    
}

tensor_t Qwen2Impl::kcache_view(size_t layer_id, size_t total_len) {
    return k_cache[layer_id]->slice(0, 0, total_len);
}
tensor_t Qwen2Impl::vcache_view(size_t layer_id, size_t total_len) {
    return v_cache[layer_id]->slice(0, 0, total_len);
}
} // namespace llaisys::models::qwen2