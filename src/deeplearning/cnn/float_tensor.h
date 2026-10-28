#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace deeplearning {

// Contiguous NCHW float tensor used by performance-sensitive CNN paths.
class FloatTensor4D {
public:
  FloatTensor4D() = default;
  FloatTensor4D(int batch, int channels, int height, int width,
                float value = 0.0f) {
    Resize(batch, channels, height, width, value);
  }

  void Resize(int batch, int channels, int height, int width,
              float value = 0.0f) {
    if (batch < 0 || channels < 0 || height < 0 || width < 0) {
      throw std::invalid_argument("FloatTensor4D dimensions must be nonnegative");
    }
    batch_ = batch;
    channels_ = channels;
    height_ = height;
    width_ = width;
    data_.assign(static_cast<std::size_t>(batch) * channels * height * width,
                 value);
  }

  void Fill(float value) { std::fill(data_.begin(), data_.end(), value); }

  bool SameShape(const FloatTensor4D &other) const {
    return batch_ == other.batch_ && channels_ == other.channels_ &&
           height_ == other.height_ && width_ == other.width_;
  }

  std::size_t Offset(int batch, int channel, int row, int column) const {
    return ((static_cast<std::size_t>(batch) * channels_ + channel) * height_ +
            row) *
               width_ +
           column;
  }

  float &operator()(int batch, int channel, int row, int column) {
    return data_[Offset(batch, channel, row, column)];
  }
  const float &operator()(int batch, int channel, int row, int column) const {
    return data_[Offset(batch, channel, row, column)];
  }

  int batch() const { return batch_; }
  int channels() const { return channels_; }
  int height() const { return height_; }
  int width() const { return width_; }
  std::size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

  float *data() { return data_.data(); }
  const float *data() const { return data_.data(); }
  std::vector<float> &values() { return data_; }
  const std::vector<float> &values() const { return data_; }

private:
  int batch_ = 0;
  int channels_ = 0;
  int height_ = 0;
  int width_ = 0;
  std::vector<float> data_;
};

} // namespace deeplearning
