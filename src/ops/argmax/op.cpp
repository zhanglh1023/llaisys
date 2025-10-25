#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    //std::cout<<max_val->dtype()<<" "<<vals->dtype()<<"\n";
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    if(max_idx->numel() != 1 || max_val->numel() != 1
        || max_idx->dtype() != LLAISYS_DTYPE_I64) {
        std::cout<<max_idx->numel()<<" "<<max_val->numel()<<" "
        <<max_idx->dtype()<<" "<<LLAISYS_DTYPE_I64<<"\n";
        TO_BE_IMPLEMENTED();
    }
        
    
    if(vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch(vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
