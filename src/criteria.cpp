#include "criteria.hpp"

double Criteria::gini(const Dataset& dataset) {    
    double p0 = static_cast<double>(dataset.target_0_count()) / dataset.size();
    double p1 = static_cast<double>(dataset.target_1_count()) / dataset.size();

    return 1.0 - (p0 * p0 + p1 * p1);
}
