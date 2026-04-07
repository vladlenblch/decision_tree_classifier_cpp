#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "tree.hpp"
#include "types.hpp"

Dataset create_dataset(size_t n_samples, size_t n_features = 4) {
  Dataset data;
  data.samples.reserve(n_samples);
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  std::uniform_int_distribution<> target_dist(0, 1);

  for (size_t i = 0; i < n_samples; i++) {
    Sample s;
    s.features.reserve(n_features);
    for (size_t j = 0; j < n_features; j++) {
      s.features.push_back(dist(gen));
    }
    s.target = target_dist(gen);
    data.samples.push_back(s);
  }
  return data;
}

static void BM_Predict_Depth5(benchmark::State& state) {
  Dataset data = create_dataset(1000);
  DecisionTree tree(5, 2, "gini");
  tree.fit(data);
  std::vector<double> sample(4, 0.5);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tree.predict(sample));
  }
}
BENCHMARK(BM_Predict_Depth5);

static void BM_Predict_Depth10(benchmark::State& state) {
  Dataset data = create_dataset(1000);
  DecisionTree tree(10, 2, "gini");
  tree.fit(data);
  std::vector<double> sample(4, 0.5);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tree.predict(sample));
  }
}
BENCHMARK(BM_Predict_Depth10);

static void BM_Fit_Gini_1000Samples(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    Dataset data = create_dataset(1000);
    state.ResumeTiming();

    DecisionTree tree(5, 2, "gini");
    tree.fit(data);
  }
}
BENCHMARK(BM_Fit_Gini_1000Samples);

static void BM_Fit_Entropy_1000Samples(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    Dataset data = create_dataset(1000);
    state.ResumeTiming();

    DecisionTree tree(5, 2, "entropy");
    tree.fit(data);
  }
}
BENCHMARK(BM_Fit_Entropy_1000Samples);

static void BM_Fit_Gini_5000Samples(benchmark::State& state) {
  Dataset data = create_dataset(5000);
  for (auto _ : state) {
    state.PauseTiming();
    Dataset data_copy = data;
    state.ResumeTiming();

    DecisionTree tree(5, 2, "gini");
    tree.fit(data_copy);
  }
}
BENCHMARK(BM_Fit_Gini_5000Samples);

static void BM_Fit_Entropy_5000Samples(benchmark::State& state) {
  Dataset data = create_dataset(5000);
  for (auto _ : state) {
    state.PauseTiming();
    Dataset data_copy = data;
    state.ResumeTiming();

    DecisionTree tree(5, 2, "entropy");
    tree.fit(data_copy);
  }
}
BENCHMARK(BM_Fit_Entropy_5000Samples);

BENCHMARK_MAIN();
