#pragma once
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
};
