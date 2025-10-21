#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, const size_t numl, const std::vector<size_t>& shape, const std::vector<int64_t>& strides) {
    for(size_t i = 0;i < numl;i++) {
        int64_t x = index[i];
        for(size_t j = 0;j < shape[1];j++) {
            out[i * strides[0] + j * strides[1]] = weight[x * strides[0] + j * strides[1]];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, const size_t numl,
                const std::vector<size_t>& shape, const std::vector<int64_t>& strides) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), 
                    reinterpret_cast<const float *>(weight), numl, shape, strides);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), numl, shape, strides);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), numl, shape, strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
