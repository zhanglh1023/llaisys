#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    size_t expected_stride = 1;

    for(size_t i = this->shape().size() - 1;i > 0;i--) {
        if((size_t)(this->strides()[i]) != expected_stride) return false;
        expected_stride *= this->shape()[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if(order.size() != this->shape().size()) throw std::runtime_error("Tensor::permute: order size uncorrect");
    if(!this->isContiguous()) throw std::runtime_error("Tensor::permute: old strides not contiguous");
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
    for(size_t i = 0;i < order.size();i++) {
        shape.push_back(this->shape()[order[i]]);
        strides.push_back(this->strides()[order[i]]);
    }
    TensorMeta meta = {this->dtype(), shape, strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    size_t old_numel = this->numel();
    if(new_numel != old_numel) throw std::runtime_error("Tensor::view: view shape has different number of elements");
    if(!this->contiguous()) throw std::runtime_error("Tensor::view: old shape not contiguous");
    std::vector<ptrdiff_t> strides;
    size_t expected_stride = 1;
    for(int i = shape.size() - 1;i >= 0;i--) {
        strides.push_back(expected_stride);
        expected_stride *= shape[i];
    }
    for(size_t i = 0;i < shape.size() / 2;i++) {
        std::swap(strides[i], strides[shape.size() - 1 - i]);
    }
    llaisys::TensorMeta meta = {this->dtype(), shape, strides};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if(dim > _meta.shape.size()) throw std::runtime_error("Slice dimension out of range: " + std::to_string(dim));
    if(start > end) throw std::runtime_error("Slice start cannot be greater than end");
    if (end > _meta.shape[dim]) {
        throw std::runtime_error("Slice end index out of range: " + std::to_string(end));
    }
    std::vector<size_t> new_shape = _meta.shape;
    std::vector<ptrdiff_t> new_strides = _meta.strides;
    new_shape[dim] = end - start;
    TensorMeta new_meta = {_meta.dtype, new_shape, new_strides};
    size_t new_offset = _offset + start * _meta.strides[dim] * this->elementSize();
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    size_t total_bytes = this->numel() * this->elementSize();

    auto& runtime = core::context().runtime();
    auto api = runtime.api();
    if(runtime.deviceType() == LLAISYS_DEVICE_CPU) {
        if(this->deviceType() == LLAISYS_DEVICE_CPU)
            api->memcpy_sync((void*)this->data(), (void*)src_, total_bytes, LLAISYS_MEMCPY_H2H);
        else api->memcpy_sync((void*)this->data(), (void*)src_, total_bytes, LLAISYS_MEMCPY_H2D);
    } else {
        if(this->deviceType() == LLAISYS_DEVICE_CPU)
            api->memcpy_sync((void*)this->data(), (void*)src_, total_bytes, LLAISYS_MEMCPY_D2H);
        else api->memcpy_sync((void*)this->data(), (void*)src_, total_bytes, LLAISYS_MEMCPY_D2D);
    }

    core::context().runtime().synchronize();
}

tensor_t Tensor::contiguous() const {
    //TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    //TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    //TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
