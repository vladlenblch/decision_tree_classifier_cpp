#pragma once
#include "types.hpp"

struct SplitResult {
    Dataset train;
    Dataset test;
};

SplitResult train_test_split(const Dataset& dataset, float test_size = 0.3, int seed = 42);
