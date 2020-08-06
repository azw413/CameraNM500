cmake_minimum_required(VERSION 3.17)
project(CameraNM500)

set(CMAKE_CXX_STANDARD 14)

include_directories("/usr/local/include/opencv4")

add_executable(CameraNM500 main.cpp)

target_link_libraries(CameraNM500 opencv_imgproc.so)
target_link_libraries(CameraNM500 opencv_imgcodecs.so)
target_link_libraries(CameraNM500 opencv_core.so)
target_link_libraries(CameraNM500 opencv_videoio.so)
target_link_libraries(CameraNM500 opencv_highgui.so)
target_link_libraries(CameraNM500 nmengine.so)