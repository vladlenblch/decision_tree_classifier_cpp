#pragma once
#include "types.hpp"

class Criteria {
public:
  static double gini(const Dataset& dataset);
  static double entropy(const Dataset& dataset);
};
