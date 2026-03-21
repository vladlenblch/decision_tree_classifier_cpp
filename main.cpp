#include <iostream>

#include "criteria.hpp"
#include "metrics.hpp"
#include "parser.hpp"
#include "split.hpp"
#include "tree.hpp"
#include "types.hpp"

int main() {
  Dataset dataset = parse_csv("data/banknotes.txt");
  SplitResult split = train_test_split(dataset, 0.2, 42);

  size_t train_0_count = split.train.target_0_count();
  size_t train_1_count = split.train.target_1_count();

  size_t test_0_count = split.test.target_0_count();
  size_t test_1_count = split.test.target_1_count();

  size_t total_0_count = train_0_count + test_0_count;
  size_t total_1_count = train_1_count + test_1_count;

  std::cout << "-------------------------------\n";
  std::cout << "Dataset size: " << dataset.size() << "\n";
  std::cout << "Dataset 0 count: " << total_0_count << "\n";
  std::cout << "Dataset 1 count: " << total_1_count << "\n";
  std::cout << "Dataset 1 to 0 ratio: " << static_cast<double>(total_1_count) / total_0_count
            << "\n";
  std::cout << "-------------------------------\n";
  std::cout << "Train size: " << split.train.size() << "\n";
  std::cout << "Train 0 count: " << train_0_count << "\n";
  std::cout << "Train 1 count: " << train_1_count << "\n";
  std::cout << "Train 1 to 0 ratio: " << static_cast<double>(train_1_count) / train_0_count << "\n";
  std::cout << "-------------------------------\n";
  std::cout << "Test size: " << split.test.size() << "\n";
  std::cout << "Test 0 count: " << test_0_count << "\n";
  std::cout << "Test 1 count: " << test_1_count << "\n";
  std::cout << "Test 1 to 0 ratio: " << static_cast<double>(test_1_count) / test_0_count << "\n";
  std::cout << "-------------------------------\n";

  for (std::string criterion : {"gini", "entropy"}) {
    DecisionTree tree(5, 2, criterion);
    tree.fit(split.train);

    std::vector<int> predictions;
    predictions.reserve(split.test.size());
    for (const Sample& sample : split.test.samples) {
      predictions.push_back(tree.predict(sample.features));
    }

    double test_accuracy = Metrics::accuracy(split.test, predictions);
    double test_precision = Metrics::precision(split.test, predictions);
    double test_recall = Metrics::recall(split.test, predictions);

    std::cout << "Criterion: " << criterion << "\n";
    std::cout << "Test Accuracy: " << test_accuracy * 100.0 << "\n";
    std::cout << "Test Precision: " << test_precision * 100.0 << "\n";
    std::cout << "Test Recall: " << test_recall * 100.0 << "\n";
    std::cout << "-------------------------------" << std::endl;
  }
}
