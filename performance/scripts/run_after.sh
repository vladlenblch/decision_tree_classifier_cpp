#!/usr/bin/env bash

set -euo pipefail

runs=10

: > performance/results/after_gini.txt
: > performance/results/after_entropy.txt

for run in $(seq 1 "$runs"); do
  {
    echo "=== run $run / $runs ==="
    ./build/tree_benchmark --benchmark_filter="^BM_Fit_Gini_5000Samples$" --benchmark_min_time=1s
    echo
  } >> performance/results/after_gini.txt

  {
    echo "=== run $run / $runs ==="
    ./build/tree_benchmark --benchmark_filter="^BM_Fit_Entropy_5000Samples$" --benchmark_min_time=1s
    echo
  } >> performance/results/after_entropy.txt
done

echo "optimized results saved to performance/results"
