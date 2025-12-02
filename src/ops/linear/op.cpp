#include "op.hpp"

#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#include "nvidia/linear_nvidia.cuh"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (bias == nullptr) {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());   
    } else {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    }

    //TO_BE_IMPLEMENTED();
    // always support cpu calculation
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias != nullptr ? bias->data() : nullptr, 
                        out->shape(), out->strides(), in->shape(), in->strides(), weight->shape(), weight->strides(),
                        in->dtype());
    }

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias != nullptr ? bias->data() : nullptr, 
                        out->shape(), out->strides(), in->shape(), in->strides(), weight->shape(), weight->strides(),
                        in->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias != nullptr ? bias->data() : nullptr, 
                            out->shape()[0], in->shape()[1], out->shape()[1],
                            in->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
