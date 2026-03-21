cmake_minimum_required(VERSION 3.10)
project(DecisionTreeClassifier)

set(CMAKE_CXX_STANDARD 23)

# ------

option(ENABLE_ASAN "Enable Address Sanitizer" ON)

if(ENABLE_ASAN)
    message(STATUS "Address Sanitizer: ENABLED")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
else()
    message(STATUS "Address Sanitizer: DISABLED")
endif()

# ------

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

# ------

enable_testing()
find_package(GTest REQUIRED)
add_executable(tree_test tests/test_main.cpp)
target_link_libraries(tree_test PRIVATE tree_lib GTest::gtest_main Threads::Threads)
add_test(NAME TreeTests COMMAND tree_test)

# ------

find_package(benchmark QUIET)

if(benchmark_FOUND)
    add_executable(tree_benchmark benchmarks/benchmark_main.cpp)
    target_link_libraries(tree_benchmark PRIVATE tree_lib benchmark::benchmark)
endif()
