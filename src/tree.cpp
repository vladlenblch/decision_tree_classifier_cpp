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
  size_t total_count_0 = dataset.target_0_count();
  size_t total_count_1 = dataset.target_1_count();
  double parent_impurity = Criterion::calculate_from_counts(total_count_0, total_count_1);

  for (size_t feature_idx = 0; feature_idx < n_features; feature_idx++) {
    std::vector<std::pair<double, unsigned int>> feature_values;
    feature_values.reserve(n_samples);

    for (const Sample& sample : dataset.samples) {
      feature_values.push_back({sample.features[feature_idx], sample.target});
    }

    std::sort(feature_values.begin(), feature_values.end());

    size_t left_count_0 = 0;
    size_t left_count_1 = 0;
    size_t right_count_0 = total_count_0;
    size_t right_count_1 = total_count_1;

    for (size_t i = 0; i < n_samples - 1; i++) {
      if (feature_values[i].second == 0) {
        left_count_0++;
        right_count_0--;
      } else {
        left_count_1++;
        right_count_1--;
      }

      if (feature_values[i].first == feature_values[i + 1].first) {
        continue;
      }

      double threshold = (feature_values[i].first + feature_values[i + 1].first) / 2.0;
      size_t n_left = i + 1;
      size_t n_right = n_samples - n_left;
      double left_impurity = Criterion::calculate_from_counts(left_count_0, left_count_1);
      double right_impurity = Criterion::calculate_from_counts(right_count_0, right_count_1);

      double children_impurity =
          (n_left * left_impurity + n_right * right_impurity) / static_cast<double>(n_samples);
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
  result.left.samples.reserve(dataset.size());
  result.right.samples.reserve(dataset.size());

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
