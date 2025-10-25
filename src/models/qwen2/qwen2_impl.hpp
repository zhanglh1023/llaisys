#pragma once

#include "llaisys/models/qwen2.h"

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


private:
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;

    llaisysDeviceType_t device;
    std::vector<int> device_ids;

    std::vector<int64_t> ctx_tokens_;

};
}