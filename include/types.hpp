#pragma once
#include <algorithm>
#include <ranges>
#include <string>
#include <vector>

struct Sample {
  std::vector<double> features;
  int target;
};

struct Dataset {
  std::vector<Sample> samples;

  void add(const Sample& sample) {
    samples.push_back(sample);
  }

  size_t size() const {
    return samples.size();
  }

  size_t target_0_count() const {
    return std::ranges::count_if(samples, [](const Sample& sample) { return sample.target == 0; });
  }

  size_t target_1_count() const {
    return size() - target_0_count();
  }
};
