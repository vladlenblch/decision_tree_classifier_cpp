#include "metrics.hpp"

#include <algorithm>
#include <ranges>

double Metrics::accuracy(const Dataset& dataset, const std::vector<unsigned int>& predictions) {
  auto indices = std::views::iota(size_t{0}, dataset.size());
  int correct = static_cast<int>(std::ranges::count_if(indices, [&](size_t i) {
    return dataset.samples[i].target == predictions[i];
  }));

  return static_cast<double>(correct) / dataset.size();
}

double Metrics::precision(const Dataset& dataset, const std::vector<unsigned int>& predictions) {
  auto indices = std::views::iota(size_t{0}, dataset.size());
  int TP = static_cast<int>(std::ranges::count_if(indices, [&](size_t i) {
    return dataset.samples[i].target == 1 && predictions[i] == 1;
  }));
  int FP = static_cast<int>(std::ranges::count_if(indices, [&](size_t i) {
    return dataset.samples[i].target == 0 && predictions[i] == 1;
  }));

  if (TP + FP == 0) {
    return 0.0;
  }
  return static_cast<double>(TP) / (TP + FP);
}

double Metrics::recall(const Dataset& dataset, const std::vector<unsigned int>& predictions) {
  auto indices = std::views::iota(size_t{0}, dataset.size());
  int TP = static_cast<int>(std::ranges::count_if(indices, [&](size_t i) {
    return dataset.samples[i].target == 1 && predictions[i] == 1;
  }));
  int FN = static_cast<int>(std::ranges::count_if(indices, [&](size_t i) {
    return dataset.samples[i].target == 1 && predictions[i] == 0;
  }));

  if (TP + FN == 0) {
    return 0.0;
  }
  return static_cast<double>(TP) / (TP + FN);
}
