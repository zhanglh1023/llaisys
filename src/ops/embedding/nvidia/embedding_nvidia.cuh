#pragma once 
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, const size_t numl,
                const std::vector<size_t>& shape, const std::vector<int64_t>& strides);
}