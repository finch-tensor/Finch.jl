cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(taco-bench)
set(CMAKE_BUILD_TYPE "release")
option(NEVA "Build benchmarks for the neva or kiwi clusters." OFF)
option(LANKA "Build benchmarks for the lanka cluster." OFF)

add_subdirectory(taco)
add_subdirectory(benchmark)

include_directories(taco taco/include benchmark/include taco/src)

file(GLOB TEST_SOURCES *.cpp *.h)

set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++14")

add_executable(taco-bench ${TEST_SOURCES} bench.h)
target_link_libraries(taco-bench benchmark::benchmark)
target_link_libraries(taco-bench taco)
# Only link stdc++fs on linux. Mac's don't need this set.
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MACOSX TRUE)
endif()
if (NOT MACOSX)
  target_link_libraries(taco-bench stdc++fs)
endif()
