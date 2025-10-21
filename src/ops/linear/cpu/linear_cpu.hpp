#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            const std::vector<size_t>& out_shape, const std::vector<ptrdiff_t>& out_stride,
            const std::vector<size_t>& in_shape, const std::vector<ptrdiff_t>& in_stride,
            const std::vector<size_t>& weight_shape, const std::vector<ptrdiff_t>& weight_stride,
            const std::vector<size_t>& bias_shape, const std::vector<ptrdiff_t>& bias_stride,
            llaisysDataType_t type);
}