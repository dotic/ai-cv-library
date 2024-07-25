cmake_minimum_required(VERSION 3.4.1)

#include_directories(../include)
#add_library(lib_opencv SHARED IMPORTED)
#set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java4.so)
#
#find_library(log-lib log)
#
#add_library(native_opencv SHARED ../ios/Classes/native_opencv.cpp)
#target_link_libraries(native_opencv lib_opencv ${log-lib})



set(OPENCV_BASE_DIR "${CMAKE_SOURCE_DIR}/src/main/jni/opencv-mobile-4.8.1-android")

set(OPENCV_INCLUDE_DIR "${OPENCV_BASE_DIR}/sdk/native/jni/include/")
set(OPENCV_STATIC_LIB_DIR "${OPENCV_BASE_DIR}/sdk/native/staticlibs/${ANDROID_ABI}")

include_directories(${OPENCV_INCLUDE_DIR})

find_library(log-lib log)

add_library(core STATIC IMPORTED)
set_target_properties(core PROPERTIES IMPORTED_LOCATION ${OPENCV_STATIC_LIB_DIR}/libopencv_core.a)

add_library(imgproc STATIC IMPORTED)
set_target_properties(imgproc PROPERTIES IMPORTED_LOCATION ${OPENCV_STATIC_LIB_DIR}/libopencv_imgproc.a)

add_library(photo STATIC IMPORTED)
set_target_properties(photo PROPERTIES IMPORTED_LOCATION ${OPENCV_STATIC_LIB_DIR}/libopencv_photo.a)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -static-openmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -static-openmp")

add_library(native_opencv
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        ../ios/Classes/native_opencv.cpp
        )

target_link_libraries(native_opencv
        core
        imgproc
        photo
        ${log-lib}
        Threads::Threads
        -lz
        )