##
## CMAKE VERSION
##
cmake_minimum_required(VERSION 3.5)

##
## PROJECT NAME
##
project(SMPL++)

##
## COMPILER SETTINGS
##
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##
## PACKAGES
##

# xtensor
find_package(xtensor REQUIRED)

# json
find_package(nlohmann_json REQUIRED)

# Torch
find_package(Torch REQUIRED)

##
## GATHER FILES
##
file(GLOB HEADER
    "SMPL++/include/*.h"
    "SMPL++/include/*.hpp"
    "SMPL++/include/definition/*.h"
    "SMPL++/include/definition/*.hpp"
    "SMPL++/include/toolbox/*.h"
    "SMPL++/include/toolbox/*.hpp"
    "SMPL++/include/smpl/*.h"
    "SMPL++/include/smpl/*.hpp"
)

file(GLOB SOURCE
    "SMPL++/src/*.c"
    "SMPL++/src/*.cpp"
    "SMPL++/src/definition/*.c"
    "SMPL++/src/definition/*.cpp"
    "SMPL++/src/toolbox/*.c"
    "SMPL++/src/toolbox/*.cpp"
    "SMPL++/src/smpl/*.c"
    "SMPL++/src/smpl/*.cpp"
)

##
## INCLUDE DIRECTORIES
##
set(SMPLXX_INCLUDE_DIRS
    "SMPL++/include"
)

set(PROJECT_INCLUDE_DIRS
    ${SMPLXX_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${xtensor_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# message("----- Include Paths -----")
# message("SMPL++: ${SMPLXX_INCLUDE_DIRS}")
# message("xtensor: ${xtensor_INCLUDE_DIRS}")
# message("Torch: ${TORCH_INCLUDE_DIRS}")
# message("-------------------------------")

##
## LIBRARIES
##
set(PROJECT_LIBRARIES
    xtensor
    ${xtensor_blas_LIBRARIES}
    nlohmann_json::nlohmann_json
    ${TORCH_LIBRARIES}
    stdc++fs                        # std::experimental::filesystem
)

# message("----- Libraries Paths -----")
# message("Torch: ${TORCH_LIBRARIES}")
# message("-------------------------------")

##
## OUTPUTS
##
add_executable(smplpp ${HEADER} ${SOURCE})
target_include_directories(smplpp PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(smplpp ${PROJECT_LIBRARIES})
