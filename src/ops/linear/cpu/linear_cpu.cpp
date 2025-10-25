#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T* bias, 
            const std::vector<size_t>& out_shape, const std::vector<ptrdiff_t>& out_stride,
            const std::vector<size_t>& in_shape, const std::vector<ptrdiff_t>& in_stride,
            const std::vector<size_t>& weight_shape, const std::vector<ptrdiff_t>& weight_stride) {
    for(size_t i = 0;i < out_shape[0];i++) {
        for(size_t j = 0;j < out_shape[1];j++) {
            float sum = 0;
            for(size_t k = 0;k < in_shape[1];k++) {
                sum += (llaisys::utils::cast<float>(in[i * in_stride[0] + k * in_stride[1]]) * 
                        llaisys::utils::cast<float>(weight[j * weight_stride[0] + k * weight_stride[1]]));
            }
            if(bias != nullptr) sum += llaisys::utils::cast<float>(bias[j]);
            out[i * out_stride[0] + j * out_stride[1]] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            const std::vector<size_t>& out_shape, const std::vector<ptrdiff_t>& out_stride,
            const std::vector<size_t>& in_shape, const std::vector<ptrdiff_t>& in_stride,
            const std::vector<size_t>& weight_shape, const std::vector<ptrdiff_t>& weight_stride,
            llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight),
                    reinterpret_cast<const float *>(bias), 
                    out_shape, out_stride,
                    in_shape, in_stride,
                    weight_shape, weight_stride);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), 
                    out_shape, out_stride,
                    in_shape, in_stride,
                    weight_shape, weight_stride);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), 
                    out_shape, out_stride,
                    in_shape, in_stride,
                    weight_shape, weight_stride);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
