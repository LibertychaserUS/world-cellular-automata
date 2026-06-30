#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "wca_kernels/dense_pair_reduce.hpp"

namespace {

constexpr std::array<char, 8> kMagic = {'W', 'C', 'A', 'D', 'P', 'R', '1', '\0'};

std::uint64_t checked_count(std::uint64_t a, std::uint64_t b, const char* label) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
    throw std::runtime_error(std::string("tensor element count overflows for ") + label);
  }
  return a * b;
}

std::vector<float> read_f32(std::ifstream& in, std::uint64_t count, const char* label) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float))) {
    throw std::runtime_error(std::string("tensor too large for host memory: ") + label);
  }
  std::vector<float> values(static_cast<std::size_t>(count));
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!in) {
    throw std::runtime_error(std::string("failed to read tensor payload: ") + label);
  }
  return values;
}

template <typename T>
T read_scalar(std::ifstream& in, const char* label) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) {
    throw std::runtime_error(std::string("failed to read ") + label);
  }
  return value;
}

void write_f32(const std::string& path, const std::vector<float>& values) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }
  out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!out) {
    throw std::runtime_error("failed to write output file: " + path);
  }
}

std::size_t parse_chunk_size(const char* raw) {
  const std::string text(raw);
  if (text.empty() || text[0] == '+' || text[0] == '-') {
    throw std::runtime_error("chunk size must be an unsigned decimal integer without a leading sign");
  }
  std::size_t parsed_chars = 0;
  const unsigned long long value = std::stoull(text, &parsed_chars);
  if (parsed_chars != text.size()) {
    throw std::runtime_error("chunk size must be a non-negative integer");
  }
  if (value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("chunk size exceeds host size_t");
  }
  return static_cast<std::size_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 4) {
      std::cerr << "usage: dense_pair_reduce_mock_cli <fixture.bin> <output.bin> [chunk_size]\n";
      return 2;
    }
    const std::size_t chunk_size = argc == 4 ? parse_chunk_size(argv[3]) : 0;

    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
      throw std::runtime_error(std::string("failed to open input file: ") + argv[1]);
    }
    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in || magic != kMagic) {
      throw std::runtime_error("fixture magic/version mismatch");
    }

    const std::uint64_t b = read_scalar<std::uint64_t>(in, "batch_size");
    const std::uint64_t c = read_scalar<std::uint64_t>(in, "center_count");
    const std::uint64_t r = read_scalar<std::uint64_t>(in, "receiver_count");
    const std::uint64_t s = read_scalar<std::uint64_t>(in, "sender_count");
    const std::uint64_t d = read_scalar<std::uint64_t>(in, "hidden_dim");
    const float residual_scale = read_scalar<float>(in, "residual_scale");
    if (b == 0 || c == 0 || r == 0 || s == 0 || d == 0) {
      throw std::runtime_error("fixture dimensions must be positive");
    }
    if (b > std::numeric_limits<std::size_t>::max() || c > std::numeric_limits<std::size_t>::max() ||
        r > std::numeric_limits<std::size_t>::max() || s > std::numeric_limits<std::size_t>::max() ||
        d > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("fixture dimensions exceed host size_t");
    }

    const std::uint64_t local_count = checked_count(checked_count(checked_count(b, c, "local"), r, "local"), d, "local");
    const std::uint64_t delta_count = checked_count(
        checked_count(checked_count(checked_count(b, c, "pair_delta"), r, "pair_delta"), s, "pair_delta"),
        d,
        "pair_delta");
    const std::uint64_t mask_count = checked_count(checked_count(b, r, "mask"), s, "mask");
    const std::uint64_t denom_count = checked_count(b, r, "denom");

    const std::vector<float> local_world = read_f32(in, local_count, "local_world");
    const std::vector<float> pair_delta = read_f32(in, delta_count, "pair_delta");
    const std::vector<float> mask = read_f32(in, mask_count, "mask");
    const std::vector<float> denom = read_f32(in, denom_count, "denom");
    if (in.peek() != std::ifstream::traits_type::eof()) {
      throw std::runtime_error("fixture contains trailing bytes");
    }

    std::vector<float> output(static_cast<std::size_t>(local_count), 0.0f);
    wca_kernels::dense_pair_reduce_cpu(
        local_world.data(),
        pair_delta.data(),
        mask.data(),
        denom.data(),
        residual_scale,
        wca_kernels::DensePairReduceShape{
            static_cast<std::size_t>(b),
            static_cast<std::size_t>(c),
            static_cast<std::size_t>(r),
            static_cast<std::size_t>(s),
            static_cast<std::size_t>(d),
        },
        chunk_size,
        output.data());
    write_f32(argv[2], output);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "dense_pair_reduce_mock_cli: " << error.what() << "\n";
    return 1;
  }
}
