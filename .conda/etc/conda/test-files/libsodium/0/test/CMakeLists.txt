cmake_minimum_required(VERSION 3.15)
project(test_libsodium C)

find_path(
    SODIUM_INCLUDE_DIRS
    NAMES sodium.h
)
if(NOT SODIUM_INCLUDE_DIRS)
  message(FATAL_ERROR "sodium.h not found")
endif()

find_library(
    SODIUM_LIBRARIES
    NAMES libsodium sodium
)

if(NOT SODIUM_LIBRARIES)
  message(FATAL_ERROR "libsodium not found")
endif()
add_library(libsodium UNKNOWN IMPORTED)
set_target_properties(libsodium PROPERTIES
  IMPORTED_LOCATION "${SODIUM_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${SODIUM_INCLUDE_DIRS}"
)

add_executable(test_version test_version.c)
target_link_libraries(test_version PRIVATE libsodium)

enable_testing()
add_test(NAME test_version COMMAND test_version)
