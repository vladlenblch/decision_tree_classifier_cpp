#include "types.hpp"
#include "parser.hpp"
#include "split.hpp"
#include <iostream>

int main() {
    Dataset dataset = parse_csv("data/banknotes.txt");
    SplitResult split = train_test_split(dataset, 0.3, 42);

    size_t train_0_count = split.train.target_0_count();
    size_t train_1_count = split.train.target_1_count();

    size_t test_0_count = split.test.target_0_count();
    size_t test_1_count = split.test.target_1_count();

    size_t total_0_count = train_0_count + test_0_count;
    size_t total_1_count = train_1_count + test_1_count;

    std::cout << "Dataset size: " << dataset.size() << std::endl;
    std::cout << "Dataset 0 count: " << total_0_count << std::endl;
    std::cout << "Dataset 1 count: " << total_1_count << std::endl;
    std::cout << "Dataset 1 to 0 ratio: " << (double)total_1_count / total_0_count << std::endl;
    std::cout << std::endl;

    std::cout << "Train size: " << split.train.size() << std::endl;
    std::cout << "Train 0 count: " << train_0_count << std::endl;
    std::cout << "Train 1 count: " << train_1_count << std::endl;
    std:: cout << "Train 1 to 0 ratio: " << (double)train_1_count / train_0_count << std::endl;
    std::cout << std::endl;

    std::cout << "Test size: " << split.test.size() << std::endl;
    std::cout << "Test 0 count: " << test_0_count << std::endl;
    std::cout << "Test 1 count: " << test_1_count << std::endl;
    std::cout << "Test 1 to 0 ratio: " << (double)test_1_count / test_0_count << std::endl;
}
