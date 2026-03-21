cmake_minimum_required(VERSION 3.10)
project(DecisionTreeClassifier)

set(CMAKE_CXX_STANDARD 23)

add_library(tree_lib STATIC
    src/parser.cpp
    src/split.cpp
    src/tree.cpp
    src/metrics.cpp
    src/criteria.cpp
)

target_include_directories(tree_lib PUBLIC include)

add_executable(DecisionTreeClassifier main.cpp)
target_link_libraries(DecisionTreeClassifier PRIVATE tree_lib)

enable_testing()

find_package(GTest REQUIRED)

add_executable(tree_test tests/test_main.cpp)

target_link_libraries(tree_test PRIVATE tree_lib GTest::gtest_main Threads::Threads)

add_test(NAME TreeTests COMMAND tree_test)
