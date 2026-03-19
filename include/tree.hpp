#pragma once
#include "types.hpp"
#include <memory>

struct TreeNode {
    bool is_leaf;
    int predicted_class;
    int feature_index;
    double threshold;
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;

    explicit TreeNode(int predicted_class_value):
        is_leaf(true), predicted_class(predicted_class_value), 
        left(nullptr), right(nullptr) {}

    TreeNode(int feature_index_value, double threshold_value, std::unique_ptr<TreeNode> left, std::unique_ptr<TreeNode> right):
        is_leaf(false), predicted_class(-1), 
        feature_index(feature_index_value), threshold(threshold_value), 
        left(std::move(left)), right(std::move(right)) {}
};

class DecisionTree {
public:
    DecisionTree(int max_depth = 5, int min_samples_split = 2);

    void fit(const Dataset& dataset);

    int predict(const std::vector<double>& sample) const;

    double score(const Dataset& dataset) const;

private:
    std::unique_ptr<TreeNode> root;
    int max_depth;
    int min_samples_split;

    std::unique_ptr<TreeNode> build_tree(const Dataset& dataset, int depth);

    struct TreeSplitResult {
        int feature_index;
        double threshold;
        double gain;
    };
    TreeSplitResult find_best_split(const Dataset& dataset);

    struct DatasetSplitResult {
        Dataset left;
        Dataset right;
    };
    DatasetSplitResult split_dataset(const Dataset& dataset, int feature_index, double threshold);

    int get_majority_class(const Dataset& dataset) const;

    int predict_sample(const std::vector<double>& sample, const TreeNode* node) const;
};
