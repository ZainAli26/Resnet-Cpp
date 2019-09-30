cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(resnet-app)

set(CMAKE_PREFIX_PATH /home/zain/libtorch)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(SRC_FILES resnet.h
    resnet.cpp)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(resnet main.cpp ${SRC_FILES})
target_link_libraries(resnet ${OpenCV_LIBS} "${TORCH_LIBRARIES}")
target_link_libraries(resnet pahottpp)
target_link_libraries(resnet pahomqtt3a)
set_property(TARGET resnet PROPERTY CXX_STANDARD 11)
