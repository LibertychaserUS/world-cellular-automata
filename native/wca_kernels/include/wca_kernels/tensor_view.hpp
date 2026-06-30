#pragma once

#include <cstddef>
#include <stdexcept>

namespace wca_kernels {

class TensorView4D {
 public:
  TensorView4D(const float* data, std::size_t b, std::size_t c, std::size_t r, std::size_t d)
      : data_(data), b_(b), c_(c), r_(r), d_(d) {
    if (data == nullptr) {
      throw std::invalid_argument("TensorView4D data must not be null");
    }
  }

  float operator()(std::size_t b, std::size_t c, std::size_t r, std::size_t d) const {
    check(b < b_ && c < c_ && r < r_ && d < d_, "TensorView4D index out of bounds");
    return data_[((b * c_ + c) * r_ + r) * d_ + d];
  }

 private:
  static void check(bool ok, const char* message) {
    if (!ok) {
      throw std::out_of_range(message);
    }
  }

  const float* data_;
  std::size_t b_;
  std::size_t c_;
  std::size_t r_;
  std::size_t d_;
};

class TensorView5D {
 public:
  TensorView5D(
      const float* data,
      std::size_t b,
      std::size_t c,
      std::size_t r,
      std::size_t s,
      std::size_t d)
      : data_(data), b_(b), c_(c), r_(r), s_(s), d_(d) {
    if (data == nullptr) {
      throw std::invalid_argument("TensorView5D data must not be null");
    }
  }

  float operator()(std::size_t b, std::size_t c, std::size_t r, std::size_t s, std::size_t d) const {
    check(b < b_ && c < c_ && r < r_ && s < s_ && d < d_, "TensorView5D index out of bounds");
    return data_[(((b * c_ + c) * r_ + r) * s_ + s) * d_ + d];
  }

 private:
  static void check(bool ok, const char* message) {
    if (!ok) {
      throw std::out_of_range(message);
    }
  }

  const float* data_;
  std::size_t b_;
  std::size_t c_;
  std::size_t r_;
  std::size_t s_;
  std::size_t d_;
};

class MutableTensorView4D {
 public:
  MutableTensorView4D(float* data, std::size_t b, std::size_t c, std::size_t r, std::size_t d)
      : data_(data), b_(b), c_(c), r_(r), d_(d) {
    if (data == nullptr) {
      throw std::invalid_argument("MutableTensorView4D data must not be null");
    }
  }

  float& operator()(std::size_t b, std::size_t c, std::size_t r, std::size_t d) {
    check(b < b_ && c < c_ && r < r_ && d < d_, "MutableTensorView4D index out of bounds");
    return data_[((b * c_ + c) * r_ + r) * d_ + d];
  }

 private:
  static void check(bool ok, const char* message) {
    if (!ok) {
      throw std::out_of_range(message);
    }
  }

  float* data_;
  std::size_t b_;
  std::size_t c_;
  std::size_t r_;
  std::size_t d_;
};

}  // namespace wca_kernels
