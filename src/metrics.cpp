#include "metrics.hpp"

double Metrics::accuracy(const Dataset& dataset, const std::vector<int>& predictions) {
    int correct = 0;

    for (size_t i = 0; i < dataset.size(); i++) {
        if (dataset.samples[i].target == predictions[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / dataset.size();
}

double Metrics::precision(const Dataset& dataset, const std::vector<int>& predictions) {
    int TP = 0, FP = 0;

    for (size_t i = 0; i < dataset.size(); i++) {
        if (dataset.samples[i].target == 1 && predictions[i] == 1) {
            TP++;
        }

        if (dataset.samples[i].target == 0 && predictions[i] == 1) {
            FP++;
        }
    }

    if (TP + FP == 0) {
        return 0.0;
    }
    return static_cast<double>(TP) / (TP + FP);
}

double Metrics::recall(const Dataset& dataset, const std::vector<int>& predictions) {
    int TP = 0, FN = 0;

    for (size_t i = 0; i < dataset.size(); i++) {
        if (dataset.samples[i].target == 1 && predictions[i] == 1) {
            TP++;
        }

        if (dataset.samples[i].target == 1 && predictions[i] == 0) {
            FN++;
        }
    }

    if (TP + FN == 0) {
        return 0.0;
    }
    return static_cast<double>(TP) / (TP + FN);
}
