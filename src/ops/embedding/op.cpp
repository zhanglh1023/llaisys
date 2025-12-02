#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#include "nvidia/embedding_nvidia.cuh"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_SAME_DTYPE(index->dtype(), LLAISYS_DTYPE_I64);

    //TO_BE_IMPLEMENTED();
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape(), weight->strides());
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape(), weight->strides());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->numel(), weight->shape(), weight->strides());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
