cmake_minimum_required(VERSION 3.1)
project(TorchBA LANGUAGES CXX)
find_package(OpenMP REQUIRED)

if(WITH_CUDA)
    enable_language(CUDA)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

SET(OPENMP_LIB OpenMP::OpenMP_CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG}")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

execute_process (
    COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PATH
)

string(STRIP ${TORCH_PATH} TORCH_PATH)
message("${TORCH_PATH}")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

include_directories(3rd/eigen)
include_directories(${TORCH_PATH}/include)
link_directories(${TORCH_PATH}/lib)
include_directories(${TORCH_PATH}/include/torch/csrc/api/include)

add_subdirectory("3rd/pybind11")

if(WITH_CUDA)
add_library(
    LMCoreKernel SHARED
    TorchLM/cpp/kernel_impl.h
    TorchLM/cpp/kernel_impl.cu
)

target_link_libraries(
    LMCoreKernel
    torch    
)
endif(WITH_CUDA)

add_library(
    LMCore SHARED
    TorchLM/cpp/kernel.h
    TorchLM/cpp/kernel_impl.h
    TorchLM/cpp/kernel.cc
    TorchLM/cpp/interface.cc
)

if(WITH_CUDA)
target_link_libraries(
    LMCore
    LMCoreKernel
    pybind11::module
    torch
    ${OPENMP_LIB}
)
set_target_properties(LMCore PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    COMPILE_DEFINITIONS WITH_CUDA=1)

else(WITH_CUDA)
target_link_libraries(
    LMCore
    pybind11::module
    torch
    ${OPENMP_LIB}
)
set_target_properties(LMCore PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif(WITH_CUDA)


add_library(
    BACore SHARED
    BAProblem/cpp/baproblem_manager.h
    BAProblem/cpp/baproblem_manager.cc
    BAProblem/cpp/interface.cc
    BAProblem/cpp/io.h
    BAProblem/cpp/io.cc
    BAProblem/cpp/torch_util.h
    BAProblem/cpp/torch_util.cc
)
target_link_libraries(
    BACore
    pybind11::module
    torch
    ${OPENMP_LIB}
)

set_target_properties(BACore PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                          SUFFIX "${PYTHON_MODULE_EXTENSION}")
