#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "criteria.hpp"
#include "metrics.hpp"
#include "tree.hpp"
#include "types.hpp"

// test 1: 100% accuracy
TEST(MetricsTest, PerfectAccuracy) {
  Dataset data;
  std::vector<unsigned int> predictions = {0, 1, 0, 1, 1};

  for (unsigned int target : {0U, 1U, 0U, 1U, 1U}) {
    Sample s;
    s.target = target;
    data.samples.push_back(s);
  }

  double acc = Metrics::accuracy(data, predictions);
  EXPECT_DOUBLE_EQ(acc, 1.0);
}

// test 2: 0% accuracy
TEST(MetricsTest, ZeroAccuracy) {
  Dataset data;
  std::vector<unsigned int> predictions = {1, 0, 1, 0, 0};

  for (unsigned int target : {0U, 1U, 0U, 1U, 1U}) {
    Sample s;
    s.target = target;
    data.samples.push_back(s);
  }

  double acc = Metrics::accuracy(data, predictions);
  EXPECT_DOUBLE_EQ(acc, 0.0);
}

// test 3: precision, TP=2, FP=1
TEST(MetricsTest, PrecisionFormula) {
  Dataset data;
  std::vector<unsigned int> predictions = {1, 1, 1, 0, 0};

  for (unsigned int target : {1U, 1U, 0U, 0U, 0U}) {
    Sample s;
    s.target = target;
    data.samples.push_back(s);
  }

  double prec = Metrics::precision(data, predictions);
  EXPECT_NEAR(prec, 2.0 / 3.0, 1e-6);
}

// test 4: precision, TP=0, FP=0
TEST(MetricsTest, PrecisionNoPositives) {
  Dataset data;
  std::vector<unsigned int> predictions = {0, 0, 0, 0, 0};

  for (unsigned int target : {0U, 1U, 0U, 0U, 0U}) {
    Sample s;
    s.target = target;
    data.samples.push_back(s);
  }

  double prec = Metrics::precision(data, predictions);
  EXPECT_DOUBLE_EQ(prec, 0.0);
}

// test 5: recall, TP=2, FN=1
TEST(MetricsTest, RecallFormula) {
  Dataset data;
  std::vector<unsigned int> predictions = {1, 0, 1, 0, 0};

  for (unsigned int target : {1U, 1U, 1U, 0U, 0U}) {
    Sample s;
    s.target = target;
    data.samples.push_back(s);
  }

  double rec = Metrics::recall(data, predictions);
  EXPECT_NEAR(rec, 2.0 / 3.0, 1e-6);
}

// test 6: recall, TP=0, FN=0
TEST(MetricsTest, RecallNoActualPositives) {
  Dataset data;
  std::vector<unsigned int> predictions = {1, 1, 1, 0, 0};

  for (unsigned int target : {0U, 0U, 0U, 0U, 0U}) {
    Sample s;
    s.target = target;
    data.samples.push_back(s);
  }

  double rec = Metrics::recall(data, predictions);
  EXPECT_DOUBLE_EQ(rec, 0.0);
}

// test 7: gini (100/0)
TEST(CriteriaTest, GiniPureNode) {
  Dataset data;
  for (int i = 0; i < 10; i++) {
    Sample s;
    s.target = 1;
    data.samples.push_back(s);
  }
  EXPECT_DOUBLE_EQ(Criteria::gini(data), 0.0);
}

// test 8: gini (50/50)
TEST(CriteriaTest, GiniMaxImpurity) {
  Dataset data;
  for (int i = 0; i < 5; i++) {
    Sample s1, s2;
    s1.target = 0;
    s2.target = 1;
    data.samples.push_back(s1);
    data.samples.push_back(s2);
  }
  EXPECT_NEAR(Criteria::gini(data), 0.5, 1e-6);
}

// test 9: entropy (100/0)
TEST(CriteriaTest, EntropyPureNode) {
  Dataset data;
  for (int i = 0; i < 10; i++) {
    Sample s;
    s.target = 0;
    data.samples.push_back(s);
  }
  EXPECT_DOUBLE_EQ(Criteria::entropy(data), 0.0);
}

// test 10: entropy (50/50)
TEST(CriteriaTest, EntropyMaxImpurity) {
  Dataset data;
  for (int i = 0; i < 5; i++) {
    Sample s1, s2;
    s1.target = 0;
    s2.target = 1;
    data.samples.push_back(s1);
    data.samples.push_back(s2);
  }
  EXPECT_NEAR(Criteria::entropy(data), 1.0, 1e-6);
}

// test 11: depth=0, always get_majority_class
TEST(TreeTest, DepthZeroReturnsMajority) {
  Dataset data;
  for (unsigned int t : {0U, 0U, 0U, 1U, 0U}) {
    Sample s;
    s.features = {1.0, 2.0, 3.0, 4.0};
    s.target = t;
    data.samples.push_back(s);
  }

  DecisionTree tree(0, 2, "gini");
  tree.fit(data);

  std::vector<double> sample = {1.0, 2.0, 3.0, 4.0};
  EXPECT_EQ(tree.predict(sample), 0);
}

// test 12: gini and entropy on synthetic dataset
TEST(TreeTest, BothCriteriaProduceValidResults) {
  Dataset data;
  for (int i = 0; i < 50; i++) {
    Sample s;
    s.features = {static_cast<double>(i)};
    s.target = (i < 25) ? 0 : 1;
    data.samples.push_back(s);
  }

  DecisionTree tree_gini(5, 2, "gini");
  DecisionTree tree_entropy(5, 2, "entropy");

  tree_gini.fit(data);
  tree_entropy.fit(data);

  std::vector<double> sample = {10.0};
  unsigned int pred_gini = tree_gini.predict(sample);
  unsigned int pred_entropy = tree_entropy.predict(sample);

  EXPECT_TRUE(pred_gini == 0 || pred_gini == 1);
  EXPECT_TRUE(pred_entropy == 0 || pred_entropy == 1);
}
