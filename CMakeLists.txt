cmake_minimum_required(VERSION 3.16)
project(option_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(pricer INTERFACE)
target_include_directories(pricer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pricer INTERFACE Threads::Threads)
target_compile_options(pricer INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>)

add_executable(pricer_cli src/main.cpp)
target_link_libraries(pricer_cli PRIVATE pricer)
set_target_properties(pricer_cli PROPERTIES OUTPUT_NAME pricer)

enable_testing()
add_executable(pricer_tests tests/test_pricer.cpp)
target_link_libraries(pricer_tests PRIVATE pricer)
add_test(NAME pricer_tests COMMAND pricer_tests)
