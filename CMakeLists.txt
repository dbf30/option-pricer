cmake_minimum_required(VERSION 3.16)
project(option_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(pricer INTERFACE)
target_include_directories(pricer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
