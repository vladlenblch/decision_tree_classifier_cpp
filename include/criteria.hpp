#pragma once
#include "types.hpp"

class Criteria {
public:
  static double gini(const Dataset& dataset);
  static double entropy(const Dataset& dataset);
};

struct GiniCriterion {
  static double calculate(const Dataset& dataset) {
    return Criteria::gini(dataset);
  }
};

struct EntropyCriterion {
  static double calculate(const Dataset& dataset) {
    return Criteria::entropy(dataset);
  }
};
