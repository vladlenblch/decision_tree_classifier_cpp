#include "tree.hpp"
#include "criteria.hpp"
#include <algorithm>
#include <iostream>

DecisionTree::DecisionTree(int max_depth, int min_samples_split, const std::string& criterion):
    root(nullptr), max_depth(max_depth), min_samples_split(min_samples_split), criterion(criterion) {}

void DecisionTree::fit(const Dataset& dataset) {
    root = build_tree(dataset, 0);
}

int DecisionTree::predict(const std::vector<double>& sample) const {
    return predict_sample(sample, root.get());
}

std::unique_ptr<TreeNode> DecisionTree::build_tree(const Dataset& dataset, int depth) {
    if (depth > max_depth) {
        return std::make_unique<TreeNode>(get_majority_class(dataset));
    }

    if (dataset.size() < static_cast<size_t>(min_samples_split)) {
        return std::make_unique<TreeNode>(get_majority_class(dataset));
    }

    if (dataset.target_0_count() == 0 || dataset.target_1_count() == 0) {
        return std::make_unique<TreeNode>(get_majority_class(dataset));
    }

    TreeSplitResult best_split = find_best_split(dataset);

    if (best_split.gain <= 0) {
        return std::make_unique<TreeNode>(get_majority_class(dataset));
    }

    DatasetSplitResult dataset_split = split_dataset(dataset, best_split.feature_index, best_split.threshold);

    auto left_child = build_tree(dataset_split.left, depth + 1);
    auto right_child = build_tree(dataset_split.right, depth + 1);

    return std::make_unique<TreeNode>(
        best_split.feature_index, 
        best_split.threshold,
        std::move(left_child),
        std::move(right_child)
    );
}

DecisionTree::TreeSplitResult DecisionTree::find_best_split(const Dataset& dataset) {
    TreeSplitResult best_split{-1, 0.0, -1.0};

    size_t n_features = dataset.samples[0].features.size();
    size_t n_samples = dataset.size();

    for (size_t feature_idx = 0; feature_idx < n_features; feature_idx++) {
        std::vector<double> values;
        values.reserve(n_samples);

        for (const Sample& sample : dataset.samples) {
            values.push_back(sample.features[feature_idx]);
        }

        std::sort(values.begin(), values.end());

        for (size_t i = 0; i < n_samples - 1; i++) {
            if (values[i] == values[i + 1]) {
                continue;
            }

            double threshold = (values[i] + values[i + 1]) / 2.0;

            DatasetSplitResult dataset_split = split_dataset(dataset, static_cast<int>(feature_idx), threshold);

            double parent_impurity = 0.0, left_impurity = 0.0, right_impurity = 0.0;
            if (criterion == "gini") {
                parent_impurity = Criteria::gini(dataset);
                left_impurity = Criteria::gini(dataset_split.left);
                right_impurity = Criteria::gini(dataset_split.right);
            } else if (criterion == "entropy") {
                parent_impurity = Criteria::entropy(dataset);
                left_impurity = Criteria::entropy(dataset_split.left);
                right_impurity = Criteria::entropy(dataset_split.right);
            }

            size_t n = dataset.size();
            size_t n_left = dataset_split.left.size();
            size_t n_right = dataset_split.right.size();

            double children_impurity = (n_left * left_impurity + n_right * right_impurity) / static_cast<double>(n);
            double gain = parent_impurity - children_impurity;

            if (gain > best_split.gain) {
                best_split = {static_cast<int>(feature_idx), threshold, gain};
            }
        }
    }
    return best_split;
}

DecisionTree::DatasetSplitResult DecisionTree::split_dataset(const Dataset& dataset, int feature_index, double threshold) {
    DatasetSplitResult result;

    for (const Sample& sample : dataset.samples) {
        if (sample.features[feature_index] <= threshold) {
            result.left.add(sample);
        } else {
            result.right.add(sample);
        }
    }
    return result;
}

int DecisionTree::get_majority_class(const Dataset& dataset) const {
    if (dataset.target_0_count() > dataset.target_1_count()) {
        return 0;
    } else {
        return 1;
    }
}

int DecisionTree::predict_sample(const std::vector<double>& sample, const TreeNode* node) const {
    if (node->is_leaf) {
        return node->predicted_class;
    }

    if (sample[node->feature_index] <= node->threshold) {
        return predict_sample(sample, node->left.get());
    } else {
        return predict_sample(sample, node->right.get());
    }
}
