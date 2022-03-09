cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ioc_mpc)

# Set C++ 17 as standard 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_PREFIX_PATH "/home/ameduri/devel/workspace/dif_ddp/extern/libtorch")

find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED CONFIG)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(eiquadprog REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# variable to store list of targets

set(all_targets)
add_library(${PROJECT_NAME} SHARED
        src/diff_qp.cpp
    )

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Includes. Add the include dependencies
target_include_directories(
    ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
  

list(APPEND all_targets ${PROJECT_NAME})

add_executable(example example/test.cpp)
add_dependencies(example ${PROJECT_NAME})
target_link_libraries(example PUBLIC ${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

list(APPEND all_targets example)

#Pybind setup
pybind11_add_module(ioc_mpc_cpp MODULE srcpy/diff_qp.cpp)
target_include_directories(ioc_mpc_cpp PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(ioc_mpc_cpp PRIVATE pybind11::module)
target_link_libraries(ioc_mpc_cpp PRIVATE  ${PROJECT_NAME})

#Install PyBind library
install(TARGETS ioc_mpc_cpp LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})