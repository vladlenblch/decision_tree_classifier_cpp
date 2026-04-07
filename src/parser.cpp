#include "parser.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

Dataset parse_csv(const std::string& filepath) {
  Dataset dataset;
  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  std::string line;

  try {
    file.open(filepath);

    while (true) {
      try {
        std::getline(file, line);
      } catch (const std::ios_base::failure&) {
        if (file.eof()) {
          break;
        }
        throw std::runtime_error("failed to read CSV: " + filepath);
      }

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
          sample.target = static_cast<int>(num);
        }
      }

      dataset.add(sample);
    }
  } catch (const std::ios_base::failure&) {
    throw std::runtime_error("failed to read CSV: " + filepath);
  }

  return dataset;
}
