# Binary Classification Decision Tree on C++

## Установка и запуск локально

```bash
# клонировать репозиторий
git clone https://github.com/vladlenblch/decision_tree_classifier_cpp
cd decision_tree_classifier_cpp

# собрать проект
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

# запустить
cd ..
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

### later

### later

## Технологический стек и требования

- `C++23`

## Структура проекта

- `.github/workflows/` - CI/CD
- `data/` – датасет
- `include/` – заголовочные файлы
- `src/` – парсинг, сплит, дерево, метрики
- `tests/` - тесты
- `main.cpp` – точка входа в программу 
- `.clang-format` - форматтер кода
- `.clang-tidy` - линтер кода
- `CMakeLists.txt` – файл сборки
- `README.md` – текстовое описание проекта
