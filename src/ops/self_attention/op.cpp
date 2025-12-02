#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#include "nvidia/flash_attention_nvidia.cuh"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    
    if(attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale,
                    attn_val->dtype(), attn_val->shape(), attn_val->strides(), 
                    q->shape(), q->strides(), k->shape(), k->strides(),
                    v->shape(), v->strides());
    }
    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    auto l_ptr = llaisys::core::context().runtime().allocateDeviceStorage(q->shape()[0] * q->shape()[1] * sizeof(float));
    auto m_ptr = llaisys::core::context().runtime().allocateDeviceStorage(q->shape()[0] * q->shape()[1] * sizeof(float));
    switch ((attn_val->deviceType()))
    {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale,
                    attn_val->dtype(), attn_val->shape(), attn_val->strides(), 
                    q->shape(), q->strides(), k->shape(), k->strides(),
                    v->shape(), v->strides());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::flash_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, 
                                    q->shape()[0], k->shape()[0], q->shape()[1], k->shape()[1], q->shape()[2], q->dtype(), l_ptr->memory(), m_ptr->memory());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
