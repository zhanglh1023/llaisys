#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            const size_t N,
            llaisysDataType_t type);
}