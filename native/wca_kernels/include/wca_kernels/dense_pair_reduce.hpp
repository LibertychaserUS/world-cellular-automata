#pragma once

#include <cstddef>

namespace wca_kernels {

struct DensePairReduceShape {
  std::size_t batch_size;
  std::size_t center_count;
  std::size_t receiver_count;
  std::size_t sender_count;
  std::size_t hidden_dim;
};

void dense_pair_reduce_cpu(
    const float* local_world,
    const float* pair_delta,
    const float* mask,
    const float* denom,
    float residual_scale,
    DensePairReduceShape shape,
    std::size_t chunk_size,
    float* output);

}  // namespace wca_kernels
