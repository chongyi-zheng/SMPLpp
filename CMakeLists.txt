##
## CMAKE VERSION
##
cmake_minimum_required(VERSION 3.0.0)

##
## PROJECT NAME
##
project(SMPL++)

##
## COMPILER SETTINGS
##
set(CMAKE_CXX_STANDARD_REQUIRED 14) # MUST BE C++ 14 FOR {XTENSOR} !!!
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

##
## PACKAGES
##

# OpenCV (not required in this version)
# find_package(OpenCV REQUIRED)

# OpenGL (not required in this version)
# find_package(OpenGL REQUIRED)
# find_package(PkgConfig REQUIRED)
# pkg_search_module(GLEW REQUIRED glew)
# pkg_search_module(GLFW REQUIRED glfw3)

# xtensor
find_package(xtensor REQUIRED)

# xtensor-io (not required in this version)
# find_package(xtensor-io REQUIRED)

# json
find_package(nlohmann_json REQUIRED)

#
# Torch
#
find_package(Torch REQUIRED)

##
## GATHER FILES
##
file(GLOB HEADER
    "include/*.h"
    "include/*.hpp"
    "include/definition/*.h"
    "include/definition/*.hpp"
    "include/toolbox/*.h"
    "include/toolbox/*.hpp"
    "include/smpl/*.h"
    "include/smpl/*.hpp"
)

file(GLOB SOURCE
    "source/*.c"
    "source/*.cpp"
    "source/definition/*.c"
    "source/definition/*.cpp"
    "source/toolbox/*.c"
    "source/toolbox/*.cpp"
    "source/smpl/*.c"
    "source/smpl/*.cpp"
)

##
## INCLUDE DIRECTORIES
##
set(SMPLXX_INCLUDE_DIRS
    "include"
)

set(PROJECT_INCLUDE_DIRS
    ${SMPLXX_INCLUDE_DIRS}
    # ${OpenCV_INCLUDE_DIRS}        (not required in this version)
    # ${OPENGL_INCLUDE_DIR}         (not required in this version)
    ${GLEW_INCLUDE_DIRS}
    ${xtensor_INCLUDE_DIRS}
    # ${xtensor_io_INCLUDE_DIRS}    (not required in this version)
    ${TORCH_INCLUDE_DIRS}
)

# message("----- Include Paths -----")
# message("SMPL++: ${SMPLXX_INCLUDE_DIRS}")
# message("OpenCV: ${OpenCV_INCLUDE_DIRS}")
# message("OpenGL: ${OPENGL_INCLUDE_DIR}")
# message("glew: ${GLEW_INCLUDE_DIRS}")
# message("xtensor: ${xtensor_INCLUDE_DIRS}")
# message("Torch: ${TORCH_INCLUDE_DIRS}")
# message("-------------------------------")

##
## LIBRARIES
##
set(PROJECT_LIBRARIES
    # ${OpenCV_LIBRARIES}           (not required in this version)
    # ${OPENGL_LIBRARIES}           (not required in this version)
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    xtensor
    ${xtensor_blas_LIBRARIES}
    # xtensor-io
    nlohmann_json::nlohmann_json
    ${TORCH_LIBRARIES}
    stdc++fs                        # std::experimental::filesystem
)

# message("----- Libraries Paths -----")
# message("OpenCV: ${OpenCV_LIBRARIES}")    (not required in this version)
# message("OpenGL: ${OPENGL_LIBRARIES}")    (not required in this version)
# message("glew: ${GLEW_LIBRARIES}")
# message("glfw: ${GLFW_LIBRARIES}")
# message("Torch: ${TORCH_LIBRARIES}")
# message("-------------------------------")

##
## OUTPUTS
##
add_executable(smplpp.out ${HEADER} ${SOURCE})
target_include_directories(smplpp.out PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(smplpp.out ${PROJECT_LIBRARIES})
