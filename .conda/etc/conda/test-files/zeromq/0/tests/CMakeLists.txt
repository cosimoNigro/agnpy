cmake_minimum_required(VERSION 3.31)
project(cfzmq LANGUAGES C)
include(CTest)

find_package(ZeroMQ REQUIRED)

add_executable(test_has test_has.c)
target_link_libraries(test_has libzmq)

add_test(NAME has_sanity COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:test_has> 0 sanity)
add_test(NAME has_curve COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:test_has> 1 curve)
add_test(NAME has_gssapi COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:test_has> 1 gssapi)
add_test(NAME has_ipc COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:test_has> 1 ipc)

if($ENV{drafts})
  add_test(NAME has_drafts COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:test_has> 1 draft)
else()
  add_test(NAME has_no_drafts COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:test_has> 0 draft)
endif()

