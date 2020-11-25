cmake_minimum_required(VERSION 2.8.3)
project(v8_ball_detector)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS
    system
    chrono
    thread
)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib
    include/v8_ball_detector/ball_detector.hpp    
)
set_target_properties(${PROJECT_NAME}_lib PROPERTIES LINKER_LANGUAGE CXX)

add_library(stuff_lib
    include/distance_weighting/distance_weighting.hpp
    include/lbp/lbp.hpp
    include/classifier_wrapper/classifier_wrapper.hpp
)
set_target_properties(stuff_lib PROPERTIES LINKER_LANGUAGE CXX)

add_executable(go
    go.cpp
)

target_link_libraries(${PROJECT_NAME}_lib
    ${OpenCV_LIBRARIES}
    Boost::system
    Boost::chrono
)

target_link_libraries(go
    ${PROJECT_NAME}_lib
    stuff_lib
    Boost::thread
)