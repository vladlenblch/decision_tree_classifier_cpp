#pragma once
#include <memory>

#include "types.hpp"

struct TreeNode {
  bool is_leaf;
  unsigned int predicted_class;
  int feature_index;
  double threshold;
  std::unique_ptr<TreeNode> left;
  std::unique_ptr<TreeNode> right;

  explicit TreeNode(unsigned int predicted_class)
      : is_leaf(true), predicted_class(predicted_class), left(nullptr), right(nullptr) {
  }

  TreeNode(
      int feature_index,
      double threshold,
      std::unique_ptr<TreeNode> left,
      std::unique_ptr<TreeNode> right
  )
      : is_leaf(false)
      , predicted_class(0)
      , feature_index(feature_index)
      , threshold(threshold)
      , left(std::move(left))
      , right(std::move(right)) {
  }
};

class DecisionTree {
public:
  DecisionTree(int max_depth = 5, int min_samples_split = 2, const std::string& criterion = "gini");
  void fit(const Dataset& dataset);
  unsigned int predict(const std::vector<double>& sample) const;

private:
  std::unique_ptr<TreeNode> root;
  int max_depth;
  int min_samples_split;
  std::string criterion;

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

  unsigned int get_majority_class(const Dataset& dataset) const;

  unsigned int predict_sample(const std::vector<double>& sample, const TreeNode* node) const;
};
