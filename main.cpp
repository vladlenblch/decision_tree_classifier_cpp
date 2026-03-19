#include "types.hpp"
#include "parser.hpp"
#include "split.hpp"
#include <iostream>

int main() {
    Dataset dataset = parse_csv("data/banknotes.txt");
    std::cout << "Dataset size: " << dataset.size() << std::endl;

    SplitResult split = train_test_split(dataset, 0.3, 42);

    std::cout << "Train size: " << split.train.size() << std::endl;
    std::cout << "Test size: " << split.test.size() << std::endl;
}
