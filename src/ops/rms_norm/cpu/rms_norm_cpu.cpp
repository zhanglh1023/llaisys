#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, const float& eps, 
            const std::vector<size_t>& shape, const std::vector<ptrdiff_t>& strides) {
    for(size_t i = 0;i < shape[0];i++) {
        float sum = 0;
        for(size_t j = 0;j < shape[1];j++) {
            float x = llaisys::utils::cast<float>(in[i * strides[0] + j * strides[1]]);
            sum += x * x;
        }
        sum /= shape[1];
        sum += eps;
        sum = sqrt(sum);
        for(size_t j = 0;j < shape[1];j++) {
            size_t idx = i * strides[0] + j * strides[1];
            float x = llaisys::utils::cast<float>(in[idx]);
            x *= llaisys::utils::cast<float>(weight[j]);
            x /= sum;
            out[idx] = llaisys::utils::cast<T>(x);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, const float& eps,
            const std::vector<size_t>& shape, const std::vector<ptrdiff_t>& strides) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), 
                    reinterpret_cast<const float *>(weight), eps, shape, strides);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), eps, shape, strides);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), eps, shape, strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
