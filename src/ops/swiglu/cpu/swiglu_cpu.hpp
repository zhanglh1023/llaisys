#pragma
#include "llaisys.h"

#include <cstddef>
#include <vector>
namespace llaisys::ops::cpu
{
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, const std::vector<size_t>& shape, 
            const std::vector<ptrdiff_t>& strides);
} // namespace