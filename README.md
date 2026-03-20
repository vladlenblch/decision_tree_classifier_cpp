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
В качестве данных был выбран датасет с характеристиками настоящих и поддельных купюр.

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

## Технологический стек и требования

- `C++17`

## Структура проекта

- `data/` – датасет
- `include/` – заголовочные файлы
- `src/` – парсинг, сплит, дерево, метрики
- `main.cpp` – точка входа в программу 
- `CMakeLists.txt` – файл сборки
- `README.md` – текстовое описание проекта
