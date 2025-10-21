#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, const float& eps,
            const std::vector<size_t>& shape, const std::vector<ptrdiff_t>& strides);
}