#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
template<typename T>
void swiglu_(T *out, const T *gate, const T *up, 
            const std::vector<size_t>& shape, const std::vector<ptrdiff_t>& strides) {
    for(size_t i = 0;i < shape[0];i++) {
        for(size_t j = 0;j < shape[1];j++) {
            size_t idx = i * strides[0] + j * strides[1];
            float gate_x = llaisys::utils::cast<float>(gate[idx]);
            float up_x = llaisys::utils::cast<float>(up[idx]);
            float x = up_x * gate_x / (1 + expf(-gate_x));
            out[idx] = llaisys::utils::cast<T>(x);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, const std::vector<size_t>& shape, 
            const std::vector<ptrdiff_t>& strides) {
    switch (type)
    {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(gate), reinterpret_cast<const float*>(up),
                    shape, strides);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t*>(gate), reinterpret_cast<const llaisys::bf16_t *>(up),
                    shape, strides);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t*>(gate), reinterpret_cast<const llaisys::fp16_t *>(up),
                    shape, strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}