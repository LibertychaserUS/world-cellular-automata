#include "wca_kernels/dense_pair_reduce.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "wca_kernels/tensor_view.hpp"

namespace wca_kernels {
namespace {

float mask_at(const float* mask, const DensePairReduceShape& shape, std::size_t b, std::size_t r, std::size_t s) {
  return mask[(b * shape.receiver_count + r) * shape.sender_count + s];
}

float denom_at(const float* denom, const DensePairReduceShape& shape, std::size_t b, std::size_t r) {
  return denom[b * shape.receiver_count + r];
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::invalid_argument(std::string("dense_pair_reduce_cpu shape product overflows for ") + label);
  }
  return left * right;
}

void validate_shape_products(const DensePairReduceShape& shape) {
  checked_mul(
      checked_mul(checked_mul(shape.batch_size, shape.center_count, "local_world"), shape.receiver_count, "local_world"),
      shape.hidden_dim,
      "local_world");
  checked_mul(
      checked_mul(
          checked_mul(
              checked_mul(shape.batch_size, shape.center_count, "pair_delta"), shape.receiver_count, "pair_delta"),
          shape.sender_count,
          "pair_delta"),
      shape.hidden_dim,
      "pair_delta");
  checked_mul(checked_mul(shape.batch_size, shape.receiver_count, "mask"), shape.sender_count, "mask");
  checked_mul(shape.batch_size, shape.receiver_count, "denom");
}

void validate_args(
    const float* local_world,
    const float* pair_delta,
    const float* mask,
    const float* denom,
    const float* output,
    const DensePairReduceShape& shape) {
  if (local_world == nullptr || pair_delta == nullptr || mask == nullptr || denom == nullptr || output == nullptr) {
    throw std::invalid_argument("dense_pair_reduce_cpu received a null tensor pointer");
  }
  if (shape.batch_size == 0 || shape.center_count == 0 || shape.receiver_count == 0 || shape.sender_count == 0 ||
      shape.hidden_dim == 0) {
    throw std::invalid_argument("dense_pair_reduce_cpu shape dimensions must be positive");
  }
  validate_shape_products(shape);
}

}  // namespace

void dense_pair_reduce_cpu(
    const float* local_world,
    const float* pair_delta,
    const float* mask,
    const float* denom,
    float residual_scale,
    DensePairReduceShape shape,
    std::size_t chunk_size,
    float* output) {
  validate_args(local_world, pair_delta, mask, denom, output, shape);
  if (!std::isfinite(residual_scale)) {
    throw std::invalid_argument("dense_pair_reduce_cpu residual_scale must be finite");
  }

  const TensorView4D local(local_world, shape.batch_size, shape.center_count, shape.receiver_count, shape.hidden_dim);
  const TensorView5D delta(
      pair_delta,
      shape.batch_size,
      shape.center_count,
      shape.receiver_count,
      shape.sender_count,
      shape.hidden_dim);
  MutableTensorView4D out(output, shape.batch_size, shape.center_count, shape.receiver_count, shape.hidden_dim);
  const std::size_t sender_chunk = chunk_size == 0 ? shape.sender_count : chunk_size;

  for (std::size_t b = 0; b < shape.batch_size; ++b) {
    for (std::size_t c = 0; c < shape.center_count; ++c) {
      for (std::size_t r = 0; r < shape.receiver_count; ++r) {
        const float denom_value = denom_at(denom, shape, b, r);
        if (!std::isfinite(denom_value) || denom_value == 0.0f) {
          throw std::invalid_argument("dense_pair_reduce_cpu denom entries must be finite and non-zero");
        }
        const float inv_denom = 1.0f / denom_value;
        for (std::size_t d = 0; d < shape.hidden_dim; ++d) {
          float acc = 0.0f;
          for (std::size_t start = 0; start < shape.sender_count; start += sender_chunk) {
            const std::size_t end =
                start + sender_chunk < shape.sender_count ? start + sender_chunk : shape.sender_count;
            for (std::size_t s = start; s < end; ++s) {
              acc += delta(b, c, r, s, d) * mask_at(mask, shape, b, r, s);
            }
          }
          if (!std::isfinite(acc)) {
            throw std::invalid_argument("dense_pair_reduce_cpu accumulated sender delta must be finite");
          }
          out(b, c, r, d) = local(b, c, r, d) + residual_scale * acc * inv_denom;
        }
      }
    }
  }
}

}  // namespace wca_kernels
