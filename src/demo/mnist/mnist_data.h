#pragma once
#include <fstream>
#include <iostream>
#include <vector>

class MnistData {
public:
  enum RC {
    SUCCESS,
    FILE_OPEN_ERROR,
  };

public:
  MnistData() = default;
  RC LoadMnistData(const std::string &train_data_file,
                   const std::string &train_label_file,
                   const std::string &test_data_file,
                   const std::string &test_label_file) {
    auto rc = ReadMnistImages(train_data_file, train_data_);
    if (rc != SUCCESS) {
      return rc;
    }
    rc = ReadMnistLabel(train_label_file, train_labels_);
    if (rc != SUCCESS) {
      return rc;
    }
    rc = ReadMnistImages(test_data_file, test_data_);
    if (rc != SUCCESS) {
      return rc;
    }
    rc = ReadMnistLabel(test_label_file, test_labels_);
    if (rc != SUCCESS) {
      return rc;
    }
    return SUCCESS;
  }
  const std::vector<std::vector<double>> &train_data() { return train_data_; }
  const std::vector<int> &train_labels() { return train_labels_; }
  const std::vector<std::vector<double>> &test_data() { return test_data_; }
  const std::vector<int> &test_labels() { return test_labels_; }
  const std::string &err_msg() { return err_msg_; }

private:
  int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
  }

  RC ReadMnistLabel(const std::string &filename, std::vector<int> &labels) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
      int magic_number = 0;
      int number_of_images = 0;
      file.read((char *)&magic_number, sizeof(magic_number));
      file.read((char *)&number_of_images, sizeof(number_of_images));
      magic_number = ReverseInt(magic_number);
      number_of_images = ReverseInt(number_of_images);

      for (int i = 0; i < number_of_images; i++) {
        unsigned char label = 0;
        file.read((char *)&label, sizeof(label));
        labels.push_back((double)label);
      }
    } else {
      err_msg_ = "File open error: " + filename;
      return FILE_OPEN_ERROR;
    }
    return SUCCESS;
  }

  RC ReadMnistImages(const std::string filename,
                     std::vector<std::vector<double>> &images) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
      int magic_number = 0;
      int number_of_images = 0;
      int n_rows = 0;
      int n_cols = 0;
      unsigned char label;
      file.read((char *)&magic_number, sizeof(magic_number));
      file.read((char *)&number_of_images, sizeof(number_of_images));
      file.read((char *)&n_rows, sizeof(n_rows));
      file.read((char *)&n_cols, sizeof(n_cols));
      magic_number = ReverseInt(magic_number);
      number_of_images = ReverseInt(number_of_images);
      n_rows = ReverseInt(n_rows);
      n_cols = ReverseInt(n_cols);

      for (int i = 0; i < number_of_images; i++) {
        std::vector<double> tp;
        for (int r = 0; r < n_rows; r++) {
          for (int c = 0; c < n_cols; c++) {
            unsigned char image = 0;
            file.read((char *)&image, sizeof(image));
            if (image != 0) {
              image = 1;
            }
            tp.push_back(image);
          }
        }
        images.push_back(tp);
      }
    } else {
      err_msg_ = "File open error: " + filename;
      return FILE_OPEN_ERROR;
    }
    return SUCCESS;
  }

private:
  std::vector<std::vector<double>> train_data_;
  std::vector<int> train_labels_;
  std::vector<std::vector<double>> test_data_;
  std::vector<int> test_labels_;
  std::string err_msg_;
};
// copy from https://www.cnblogs.com/ppDoo/p/13261258.html
