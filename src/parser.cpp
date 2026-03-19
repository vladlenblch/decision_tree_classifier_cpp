#include "parser.hpp"
#include <fstream>
#include <sstream>

Dataset parse_csv(const std::string& filepath) {
    Dataset dataset;
    std::ifstream file(filepath);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        Sample sample;

        for (int i = 0; i < 5; i++) {
            if (!std::getline(ss, value, ',')) {
                break;
            }

            double num = std::stod(value);

            if (i < 4) {
                sample.features.push_back(num);
            } else {
                sample.target = (int)num;
            }
        }

        dataset.add(sample);
    }

    return dataset;
}
