#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
            const size_t N, const size_t K, const float eps,
            llaisysDataType_t type);
}