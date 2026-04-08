# Binary Classification Decision Tree on C++

## Установка и запуск локально

```bash
# клонировать репозиторий
git clone https://github.com/vladlenblch/decision_tree_classifier_cpp
cd decision_tree_classifier_cpp

# собрать проект
mkdir build
cd build
# сборка с проверкой памяти
cmake .. -DENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug
# сборка без проверки памяти (для бенчмарков)
cmake .. -DENABLE_ASAN=OFF -DCMAKE_BUILD_TYPE=Release
make

# запустить
cd ..
# запустить тесты
./build/tree_test
# запустить бенчмарки
./build/tree_benchmark --benchmark_min_time=1s
# запустить дерево 
./build/DecisionTreeClassifier
```

## О проекте

Дерево решений, решающее проблему бинарной классификации. <br>
В качестве данных был выбран датасет с характеристиками настоящих и поддельных купюр, состоящий только из числовых признаков.

### Датасет

- `variance` - дисперсия вейвлет-преобразования
- `skewness` - асимметрия
- `kurtosis` - эксцесс
- `entropy` - энтропия изображения
- `target` - класс купюры

### Гиперпараметры

- `max_depth` - максимальная глубина дерева
- `min_samples_split` - минимальное количество объектов в узле для разделения
- `criterion` - Джини или Энтропия

### Результаты выполнения программы

```zsh
-------------------------------
Dataset size: 1372
Dataset 0 count: 762
Dataset 1 count: 610
Dataset 1 to 0 ratio: 0.800525
-------------------------------
Train size: 1097
Train 0 count: 609
Train 1 count: 488
Train 1 to 0 ratio: 0.801314
-------------------------------
Test size: 275
Test 0 count: 153
Test 1 count: 122
Test 1 to 0 ratio: 0.797386
-------------------------------
Criterion: gini
Test Accuracy: 97.0909
Test Precision: 95.2381
Test Recall: 98.3607
-------------------------------
Criterion: entropy
Test Accuracy: 98.5455
Test Precision: 99.1667
Test Recall: 97.541
-------------------------------
```

### Результаты выполнения тестов

```zsh
[==========] Running 12 tests from 3 test suites.
[----------] Global test environment set-up.
[----------] 6 tests from MetricsTest
[ RUN      ] MetricsTest.PerfectAccuracy
[       OK ] MetricsTest.PerfectAccuracy (0 ms)
[ RUN      ] MetricsTest.ZeroAccuracy
[       OK ] MetricsTest.ZeroAccuracy (0 ms)
[ RUN      ] MetricsTest.PrecisionFormula
[       OK ] MetricsTest.PrecisionFormula (0 ms)
[ RUN      ] MetricsTest.PrecisionNoPositives
[       OK ] MetricsTest.PrecisionNoPositives (0 ms)
[ RUN      ] MetricsTest.RecallFormula
[       OK ] MetricsTest.RecallFormula (0 ms)
[ RUN      ] MetricsTest.RecallNoActualPositives
[       OK ] MetricsTest.RecallNoActualPositives (0 ms)
[----------] 6 tests from MetricsTest (0 ms total)

[----------] 4 tests from CriteriaTest
[ RUN      ] CriteriaTest.GiniPureNode
[       OK ] CriteriaTest.GiniPureNode (0 ms)
[ RUN      ] CriteriaTest.GiniMaxImpurity
[       OK ] CriteriaTest.GiniMaxImpurity (0 ms)
[ RUN      ] CriteriaTest.EntropyPureNode
[       OK ] CriteriaTest.EntropyPureNode (0 ms)
[ RUN      ] CriteriaTest.EntropyMaxImpurity
[       OK ] CriteriaTest.EntropyMaxImpurity (0 ms)
[----------] 4 tests from CriteriaTest (0 ms total)

[----------] 2 tests from TreeTest
[ RUN      ] TreeTest.DepthZeroReturnsMajority
[       OK ] TreeTest.DepthZeroReturnsMajority (0 ms)
[ RUN      ] TreeTest.BothCriteriaProduceValidResults
[       OK ] TreeTest.BothCriteriaProduceValidResults (3 ms)
[----------] 2 tests from TreeTest (4 ms total)

[----------] Global test environment tear-down
[==========] 12 tests from 3 test suites ran. (4 ms total)
[  PASSED  ] 12 tests.
```

### Результаты бенчмарков

```zsh
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x10)
Load Average: 2.47, 2.69, 2.51
---------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations
---------------------------------------------------------------------
BM_Predict_Depth5                6.94 ns         6.94 ns    101503705
BM_Predict_Depth10               17.9 ns         17.9 ns     38933230
BM_Fit_Gini_1000Samples     385308896 ns    385301500 ns            2
BM_Fit_Entropy_1000Samples  370184875 ns    370163500 ns            2
BM_Fit_Gini_5000Samples    1.2683e+10 ns   1.2682e+10 ns            1
BM_Fit_Entropy_5000Samples 1.2904e+10 ns   1.2900e+10 ns            1
```

### Результаты оптимизации

Средние CPU-значения по 10 прогонам бенчмарков на обучении датасета на 5000 объектов с обоими критериями до и после оптимизации:

| Benchmark | Before | After | Speedup |
| --- | --- | --- | --- |
| `BM_Fit_Gini_5000Samples` | `1.43744e+10 ns` | `5.739651e+06 ns` | `~2500x` |
| `BM_Fit_Entropy_5000Samples` | `1.45e+10 ns` | `6.654837e+06 ns` | `~2200x` |

Оптимизация была сделана в `find_best_split()`: теперь вместо физического разбиения объектов в ноде для каждого трешхолда просто считаются критерии разбития по нерабитому датасету, а физический сплит вызывается единожды - только в конце, когда нашли лучший сплит.

### CI/CD

Проект использует Github Actions для автоматической проверки кода:
1) `git push` - при пуше начинается пайплайн
2) `test-with-asan` - Unit-тесты, проверка на утечки памяти с помощью `Adress Sanitizer`
3) `benchmarks` - замеры производительности на датасетах разного размера (без `ASan` для производительности). Требует успешного прохождения тестов

## Технологический стек и требования

- `C++23`

## Структура проекта

- `.github/workflows/` - CI/CD
- `benchmarks` - бенчмарки
- `include/` – заголовочные файлы
- `src/` – парсинг, сплит, дерево, метрики
- `tests/` - тесты
- `banknotes.txt` - датасет
- `main.cpp` – точка входа в программу 
- `.clang-format` - форматтер кода
- `.clang-tidy` - линтер кода
- `CMakeLists.txt` – файл сборки
- `README.md` – текстовое описание проекта
