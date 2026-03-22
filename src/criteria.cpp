#include "criteria.hpp"

#include <cmath>

double Criteria::gini(const Dataset& dataset) {
  double p0 = static_cast<double>(dataset.target_0_count()) / dataset.size();
  double p1 = static_cast<double>(dataset.target_1_count()) / dataset.size();
  double gini = 1.0 - (p0 * p0 + p1 * p1);

  return gini;
}

double Criteria::entropy(const Dataset& dataset) {
  double p0 = static_cast<double>(dataset.target_0_count()) / dataset.size();
  double p1 = static_cast<double>(dataset.target_1_count()) / dataset.size();
  double entropy = 0.0;

  if (p0 > 0.0) {
    entropy -= p0 * std::log2(p0);
  }

  if (p1 > 0.0) {
    entropy -= p1 * std::log2(p1);
  }

  return entropy;
}
