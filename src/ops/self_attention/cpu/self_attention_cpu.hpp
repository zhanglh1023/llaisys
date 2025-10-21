#pragma
#include "llaisys.h"

#include <cstddef>
#include <vector>
namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, const float scale,
                    llaisysDataType_t type, const std::vector<size_t>& attn_val_shape, const std::vector<ptrdiff_t>& attn_val_strides, 
                    const std::vector<size_t>& q_shape, const std::vector<ptrdiff_t>& q_strides,
                    const std::vector<size_t>& k_shape, const std::vector<ptrdiff_t>& k_strides,
                    const std::vector<size_t>& v_shape, const std::vector<ptrdiff_t>& v_strides);
}
