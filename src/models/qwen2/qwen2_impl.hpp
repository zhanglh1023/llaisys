#pragma once

#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"

#include <vector>
namespace llaisys::models::qwen2 {

struct Qwen2Impl {
public:
    Qwen2Impl(const LlaisysQwen2Meta& meta_, 
            const llaisysDeviceType_t& device_,
            const int* device_ids_,
            const int ndevice_
            );
    ~Qwen2Impl();
    
    Qwen2Impl(const Qwen2Impl&) = delete;
    Qwen2Impl& operator=(const Qwen2Impl&) = delete;
    Qwen2Impl(const Qwen2Impl&&) = delete;
    Qwen2Impl& operator=(const Qwen2Impl&&) = delete;

    LlaisysQwen2Weights* getWeight() {return &weights;}

    int64_t forward(const int64_t *token_ids, size_t ntoken, size_t pos_base);
    int64_t prifill(const int64_t *token_ids, size_t ntoken);

    int64_t decode_one(const int64_t token_id);

private:
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;

    llaisysDeviceType_t device;
    std::vector<int> device_ids;

    std::vector<int64_t> ctx_tokens_;
    
    // kv cache
    bool use_kvcache = true;

    void reset_cache();
    void ensure_kvcache_capacity(size_t layer_id, size_t ntoken, llaisysDataType_t dtype, 
                            llaisysDeviceType_t devtype, int devid);
    void append_kv(size_t layer_id, const tensor_t& k, const tensor_t& v, size_t pos);
    tensor_t kcache_view(size_t layer_id, size_t total_len);
    tensor_t vcache_view(size_t layer_id, size_t total_len);
    size_t cache_len;
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
};
}