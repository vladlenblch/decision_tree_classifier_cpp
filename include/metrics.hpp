#pragma once
#include "types.hpp"

class Metrics {
public:
  static double accuracy(const Dataset& dataset, const std::vector<unsigned int>& predictions);
  static double precision(const Dataset& dataset, const std::vector<unsigned int>& predictions);
  static double recall(const Dataset& dataset, const std::vector<unsigned int>& predictions);
};
