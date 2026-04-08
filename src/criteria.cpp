#include "criteria.hpp"

#include <cmath>

double Criteria::gini(const Dataset& dataset) {
  return gini_from_counts(dataset.target_0_count(), dataset.target_1_count());
}

double Criteria::entropy(const Dataset& dataset) {
  return entropy_from_counts(dataset.target_0_count(), dataset.target_1_count());
}

double Criteria::gini_from_counts(size_t count_0, size_t count_1) {
  size_t total = count_0 + count_1;
  double p0 = static_cast<double>(count_0) / total;
  double p1 = static_cast<double>(count_1) / total;
  double gini = 1.0 - (p0 * p0 + p1 * p1);

  return gini;
}

double Criteria::entropy_from_counts(size_t count_0, size_t count_1) {
  size_t total = count_0 + count_1;
  double p0 = static_cast<double>(count_0) / total;
  double p1 = static_cast<double>(count_1) / total;
  double entropy = 0.0;

  if (p0 > 0.0) {
    entropy -= p0 * std::log2(p0);
  }

  if (p1 > 0.0) {
    entropy -= p1 * std::log2(p1);
  }

  return entropy;
}
