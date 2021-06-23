cmake_minimum_required(VERSION 3.16)
project(_gym_ignition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Scenario COMPONENTS Gazebo REQUIRED)
find_package(pybind11 2.3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Set flags for OpenMP
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed -fopenmp -w -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

message(STATUS "[gym_ignition] prefix path ${CMAKE_PREFIX_PATH}")
message(STATUS "[gym_ignition] including directory ${GYM_IGN_ENVIRONMENT_INCLUDE_PATH}")

set(SOURCE_DIR gym_ignition/env)
set(INCLUDE_DIR gym_ignition/env ${GYM_IGN_ENVIRONMENT_INCLUDE_PATH})

set(INCLUDE_DIR ${INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})

# add the Pybind model to access Environment methods
pybind11_add_module(_gym_ignition ${SOURCE_DIR}/gym_ignition.cpp)

target_link_libraries(_gym_ignition PRIVATE
ScenarioGazebo::ScenarioGazebo
ScenarioGazebo::GazeboSimulator
yaml-cpp)
target_include_directories(_gym_ignition PUBLIC ${INCLUDE_DIR})
target_compile_options(_gym_ignition PRIVATE -mtune=native -fPIC -O3)

message(STATUS "[gym_ignition] CMAKE INSTALL PREFIX ${CMAKE_INSTALL_PREFIX}")
message(STATUS "[gym_ignition] EXECUTABLES INSTALLED TO ${CMAKE_INSTALL_BINDIR}")
message(STATUS "[gym_ignition] LIBRARIES INSTALLED TO ${CMAKE_INSTALL_LIBDIR}")

# TODO: Make a debug executable
if(CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "[gym_ignition] BUILDING THE DEBUG EXECUTABLE")
    find_package(PythonLibs REQUIRED)

    # add_executable(debug_app ${SOURCE_DIR}/debug_app.cpp)
    target_link_libraries(debug_app PRIVATE
    ScenarioGazebo::ScenarioGazebo
    ScenarioGazebo::GazeboSimulator
    yaml-cpp
    ${PYTHON_LIBRARIES})
    target_include_directories(debug_app PUBLIC ${SOURCE_DIR} ${GYM_IGN_ENVIRONMENT_INCLUDE_PATH} ${EIGEN3_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
    target_compile_options(debug_app PRIVATE -mtune=native -fPIC -g -O0)
    
endif()