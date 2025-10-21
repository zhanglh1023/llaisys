#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, llaisysDataType_t type, const float& theta, 
        const std::vector<size_t>& shape, const std::vector<ptrdiff_t>& strides);
}