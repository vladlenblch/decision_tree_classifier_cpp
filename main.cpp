#include "types.hpp"
#include "parser.hpp"
#include "split.hpp"
#include "tree.hpp"
#include <iostream>

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
    std::cout << "Dataset 1 to 0 ratio: " << static_cast<double>(total_1_count) / total_0_count << "\n";
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
    std::cout << "-------------------------------" << std::endl;

    DecisionTree tree(5, 2);
    tree.fit(split.train);

    double train_score = tree.score(split.train);
    double test_score = tree.score(split.test);

    std::cout << "Train Accuracy: " << train_score * 100.0 << "\n";
    std::cout << "Test Accuracy: " << test_score * 100.0 << "\n";
}
