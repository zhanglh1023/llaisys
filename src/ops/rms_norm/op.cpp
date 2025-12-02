#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#include "nvidia/rms_norm_nvidia.cuh"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMS_NORM: all tensors must be contiguous.");
    
    
    //TO_BE_IMPLEMENTED();
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), eps, in->shape(), in->strides());
    }

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), eps, in->shape(), in->strides());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), in->shape()[0], in->shape()[1], eps, in->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
