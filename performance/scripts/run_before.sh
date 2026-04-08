#!/usr/bin/env bash

set -euo pipefail

runs=10

: > performance/results/before_gini.txt
: > performance/results/before_entropy.txt

for run in $(seq 1 "$runs"); do
  {
    echo "=== run $run / $runs ==="
    ./build/tree_benchmark --benchmark_filter="^BM_Fit_Gini_5000Samples$" --benchmark_min_time=1s
    echo
  } >> performance/results/before_gini.txt

  {
    echo "=== run $run / $runs ==="
    ./build/tree_benchmark --benchmark_filter="^BM_Fit_Entropy_5000Samples$" --benchmark_min_time=1s
    echo
  } >> performance/results/before_entropy.txt
done

echo "baseline results saved to performance/results"
