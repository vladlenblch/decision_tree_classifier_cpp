#include "types.hpp"
#include "parser.hpp"
#include <iostream>

int main() {
    Dataset dataset = parse_csv("data/banknotes.txt");
    std::cout << "Dataset size: " << dataset.size() << std::endl;
}
