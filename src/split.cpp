#include "split.hpp"
#include <random>
#include <algorithm>

SplitResult train_test_split(const Dataset& dataset, float test_size, int seed) {
    SplitResult result;
    
    std::vector<size_t> class_0_indices;
    std::vector<size_t> class_1_indices;

    for (size_t i = 0; i < dataset.samples.size(); i++) {
        if (dataset.samples[i].target == 0) {
            class_0_indices.push_back(i);
        } else {
            class_1_indices.push_back(i);
        }
    }

    std::mt19937 generate_seed(seed);

    std::shuffle(class_0_indices.begin(), class_0_indices.end(), generate_seed);
    std::shuffle(class_1_indices.begin(), class_1_indices.end(), generate_seed);

    size_t train_size_0 = (size_t)(class_0_indices.size() * (1 - test_size));
    size_t train_size_1 = (size_t)(class_1_indices.size() * (1 - test_size));

    for (size_t i = 0; i < class_0_indices.size(); i++) {
        const Sample& sample = dataset.samples[class_0_indices[i]];
        if (i < train_size_0) {
            result.train.add(sample);
        } else {
            result.test.add(sample);
        }
    }

    for (size_t i = 0; i < class_1_indices.size(); i++) {
        const Sample& sample = dataset.samples[class_1_indices[i]];
        if (i < train_size_1) {
            result.train.add(sample);
        } else {
            result.test.add(sample);
        }
    }

    std::shuffle(result.train.samples.begin(), result.train.samples.end(), generate_seed);
    std::shuffle(result.test.samples.begin(), result.test.samples.end(), generate_seed);

    return result;
}
