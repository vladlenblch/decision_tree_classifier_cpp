#pragma once
#include "types.hpp"

class Criteria {
public:
  static double gini(const Dataset& dataset);
  static double entropy(const Dataset& dataset);
  static double gini_from_counts(size_t count_0, size_t count_1);
  static double entropy_from_counts(size_t count_0, size_t count_1);
};

struct GiniCriterion {
  static double calculate(const Dataset& dataset) {
    return Criteria::gini(dataset);
  }

  static double calculate_from_counts(size_t count_0, size_t count_1) {
    return Criteria::gini_from_counts(count_0, count_1);
  }
};

struct EntropyCriterion {
  static double calculate(const Dataset& dataset) {
    return Criteria::entropy(dataset);
  }

  static double calculate_from_counts(size_t count_0, size_t count_1) {
    return Criteria::entropy_from_counts(count_0, count_1);
  }
};
