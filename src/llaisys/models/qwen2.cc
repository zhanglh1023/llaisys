#include "llaisys/models/qwen2.h"

#include "../../models/qwen2/qwen2_impl.hpp"


#include <memory>

__C {
    struct LlaisysQwen2Model {
        std::unique_ptr<llaisys::models::qwen2::Qwen2Impl> impl;
    };
    
    struct LlaisysQwen2Model* llaisysQwen2ModelCreate(
            const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice
    ) {
        try {
            LlaisysQwen2Model *model = new LlaisysQwen2Model();
            model->impl = std::make_unique<llaisys::models::qwen2::Qwen2Impl>(*meta, device, device_ids, ndevice);
            return model;
        } catch(const std::bad_alloc&) {
            return nullptr;
        } catch(...) {
            return nullptr;
        }
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        if(model == nullptr || model->impl == nullptr) return nullptr;
        return model->impl->getWeight();
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
        if(model == nullptr || model->impl == nullptr) return -1;
        try {
            return model->impl->prifill(token_ids, ntoken);
        } catch(...) {
            return -1;
        }
    }

    int64_t llaisysQwen2ModelForwardOne(struct LlaisysQwen2Model *model, int64_t token_id) {
        if(model == nullptr || model->impl == nullptr) return -1;
        try {
            return model->impl->decode_one(token_id);
        } catch(...) {
            return -1;
        }
    }


}