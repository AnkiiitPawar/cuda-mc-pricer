cmake_minimum_required(VERSION 3.20)
project(cuda_mc_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA is optional: without nvcc the GPU entry points are stubs and the CPU path
# (and its tests) still build and run. CUDA must be enabled before any target is
# created so that targets pick up CMAKE_CUDA_ARCHITECTURES.
option(MCP_USE_CUDA "Build the CUDA kernels if a CUDA compiler is found" ON)
set(MCP_HAVE_CUDA OFF)
if(MCP_USE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Remember on the first configure whether the user chose architectures, because
    # enable_language() fills in a default of its own.
    if(NOT DEFINED MCP_USER_CUDA_ARCHS)
      if(DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED ENV{CUDAARCHS})
        set(MCP_USER_CUDA_ARCHS ON CACHE INTERNAL "")
      else()
        set(MCP_USER_CUDA_ARCHS OFF CACHE INTERNAL "")
      endif()
    endif()
    enable_language(CUDA)
    if(NOT MCP_USER_CUDA_ARCHS)
      # Turing to Ada (sm_89 = RTX 40 series). CUDA 13 dropped Volta, so sm_70 is
      # only added for older compilers.
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
        list(PREPEND CMAKE_CUDA_ARCHITECTURES 70)
      endif()
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(MCP_HAVE_CUDA ON)
  endif()
endif()

add_library(mcp src/model.cpp src/cpu.cpp)
target_include_directories(mcp PUBLIC include)
if(MCP_HAVE_CUDA)
  target_sources(mcp PRIVATE src/gpu.cu)
  message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}, architectures ${CMAKE_CUDA_ARCHITECTURES}")
else()
  target_sources(mcp PRIVATE src/gpu_stub.cpp)
  message(STATUS "CUDA disabled or not found: building the CPU path only")
endif()

add_executable(mc_pricer app/main.cpp)
target_link_libraries(mc_pricer PRIVATE mcp)

enable_testing()
add_executable(test_cpu tests/test_cpu.cpp)
target_link_libraries(test_cpu PRIVATE mcp)
add_test(NAME cpu COMMAND test_cpu)

add_executable(test_gpu tests/test_gpu.cpp)
target_link_libraries(test_gpu PRIVATE mcp)
add_test(NAME gpu COMMAND test_gpu)
set_tests_properties(gpu PROPERTIES SKIP_RETURN_CODE 77)
