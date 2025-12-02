#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
            const float theta, const size_t seq_len, const size_t nhead, const size_t dim,
            llaisysDataType_t type);
}