
cmake_minimum_required(VERSION 3.12)

project(exe C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_library(pico-tflmicro-wrapped STATIC)

target_include_directories(pico-tflmicro-wrapped
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/src
)

set_target_properties(
  pico-tflmicro-wrapped
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_link_libraries(
  pico-tflmicro-wrapped
  pico-tflmicro
)

target_sources(pico-tflmicro
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/src/wrapped.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/wrapped.h
)


add_executable(exe "")

target_include_directories(exe
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  exe
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(exe
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/empty.c
)

target_link_libraries(
  exe
  pico-tflmicro-wrapped
)
