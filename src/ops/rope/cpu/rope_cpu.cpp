#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, const float& theta, const std::vector<size_t>& shape, 
        const std::vector<ptrdiff_t>& strides) {
    
    std::vector<std::vector<float> > cos_phi(shape[0], std::vector<float>(shape[2] / 2));
    std::vector<std::vector<float> > sin_phi(shape[0], std::vector<float>(shape[2] / 2));
    
    for(size_t i = 0;i < shape[0];i++) {
        for(size_t j = 0;j < shape[2] / 2;j++) {
            float phi = pos_ids[i];
            phi /= std::pow(theta, 2.0 * j / shape[2]);
            cos_phi[i][j] = std::cos(phi);
            sin_phi[i][j] = std::sin(phi);
        }
    }
    
    for(size_t i = 0;i < shape[0];i++) {
        for(size_t j = 0;j < shape[1];j++) {
            for(size_t k = 0;k < shape[2] / 2;k++) {
                size_t idx_a = i * strides[0] + j * strides[1] + k * strides[2];
                size_t idx_b = i * strides[0] + j * strides[1] + (k + shape[2] / 2) * strides[2];
                float a = llaisys::utils::cast<float>(in[idx_a]);
                float b = llaisys::utils::cast<float>(in[idx_b]);
                out[idx_a] = llaisys::utils::cast<T>(a * cos_phi[i][k] - b * sin_phi[i][k]);
                out[idx_b] = llaisys::utils::cast<T>(b * cos_phi[i][k] + a * sin_phi[i][k]); 
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, llaisysDataType_t type, const float& theta, 
        const std::vector<size_t>& shape, const std::vector<ptrdiff_t>& strides) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), 
                pos_ids, theta, shape, strides);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                pos_ids, theta, shape, strides);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                pos_ids, theta, shape, strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
