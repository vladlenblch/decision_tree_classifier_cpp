cmake_minimum_required(VERSION 3.10)
project(DecisionTreeClassifier)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(
    DecisionTreeClassifier
    main.cpp
    src/parser.cpp
    src/split.cpp
)
