#include "tree.hpp"

#include <algorithm>
#include <iostream>

template <typename Criterion>
DecisionTree<Criterion>::DecisionTree(int max_depth, int min_samples_split)
    : root(nullptr), max_depth(max_depth), min_samples_split(min_samples_split) {
}

template <typename Criterion>
void DecisionTree<Criterion>::fit(const Dataset& dataset) {
  root = build_tree(dataset, 0);
}

template <typename Criterion>
unsigned int DecisionTree<Criterion>::predict(const std::vector<double>& sample) const {
  return predict_sample(sample, root.get());
}

template <typename Criterion>
std::unique_ptr<TreeNode> DecisionTree<Criterion>::build_tree(const Dataset& dataset, int depth) {
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

  DatasetSplitResult dataset_split =
      split_dataset(dataset, best_split.feature_index, best_split.threshold);

  std::unique_ptr<TreeNode> left_child = build_tree(dataset_split.left, depth + 1);
  std::unique_ptr<TreeNode> right_child = build_tree(dataset_split.right, depth + 1);

  return std::make_unique<TreeNode>(
      best_split.feature_index, best_split.threshold, std::move(left_child), std::move(right_child)
  );
}

template <typename Criterion>
typename DecisionTree<Criterion>::TreeSplitResult DecisionTree<Criterion>::find_best_split(
    const Dataset& dataset
) {
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

      DatasetSplitResult dataset_split =
          split_dataset(dataset, static_cast<int>(feature_idx), threshold);

      double parent_impurity = Criterion::calculate(dataset);
      double left_impurity = Criterion::calculate(dataset_split.left);
      double right_impurity = Criterion::calculate(dataset_split.right);

      size_t n = dataset.size();
      size_t n_left = dataset_split.left.size();
      size_t n_right = dataset_split.right.size();

      double children_impurity =
          (n_left * left_impurity + n_right * right_impurity) / static_cast<double>(n);
      double gain = parent_impurity - children_impurity;

      if (gain > best_split.gain) {
        best_split = {static_cast<int>(feature_idx), threshold, gain};
      }
    }
  }
  return best_split;
}

template <typename Criterion>
typename DecisionTree<Criterion>::DatasetSplitResult DecisionTree<Criterion>::split_dataset(
    const Dataset& dataset, int feature_index, double threshold
) {
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

template <typename Criterion>
unsigned int DecisionTree<Criterion>::get_majority_class(const Dataset& dataset) const {
  if (dataset.target_0_count() > dataset.target_1_count()) {
    return 0;
  } else {
    return 1;
  }
}

template <typename Criterion>
unsigned int DecisionTree<Criterion>::predict_sample(
    const std::vector<double>& sample, const TreeNode* node
) const {
  if (node->is_leaf) {
    return node->predicted_class;
  }

  if (sample[node->feature_index] <= node->threshold) {
    return predict_sample(sample, node->left.get());
  } else {
    return predict_sample(sample, node->right.get());
  }
}

template class DecisionTree<GiniCriterion>;
template class DecisionTree<EntropyCriterion>;
