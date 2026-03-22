#pragma once
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
    size_t count = 0;
    for (const Sample& sample : samples) {
      if (sample.target == 0) {
        count++;
      }
    }
    return count;
  }

  size_t target_1_count() const {
    return size() - target_0_count();
  }
};
