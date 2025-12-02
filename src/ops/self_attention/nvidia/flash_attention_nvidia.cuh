#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::nvidia {
void flash_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
            const float scale, const size_t q_len, const size_t kv_len, const size_t nhead, const size_t nkv_head, const size_t dim,
            llaisysDataType_t type, std::byte *l, std::byte *m);
}