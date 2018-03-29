cmake_minimum_required(VERSION 2.6)

PROJECT(checkbitexts)
SET(USERHOME "$ENV{HOME}")
#if you don't want the full compiler output, remove the following line
SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_CXX_FLAGS " -Wno-deprecated -Wunused-local-typedefs -Wsign-compare " CACHE STRING "Flags used by the C++ compiler" FORCE )
#add_definitions(-DCMAKE_INSTALL_PREFIX=YOUR INSTALL DIR HERE!!!)
add_definitions(-DIBPP_LINUX -DIBPP_GCC)
# include_directories(YOUR INCLUDE DIR HERE!!!)
# link_directories(YOUR LIB DIR HERE!!!)
#include_directories(/DEV/Tools/boost_1_61_0/include)
#link_directories(/DEV/Tools/boost_1_61_0/lib)
#link_directories(${USERHOME}/Tools/myLibs/icu/lib)
#include_directories(${USERHOME}/Tools/myLibs/icu/include)
include_directories(${USERHOME}/Tools/MultiVec/include)
link_directories(${USERHOME}/Tools/MultiVec/lib/)
#SET(CMAKE_CXX_FLAGS "-Wunused-local-typedefs")
#add definitions, compiler switches, etc.
ADD_DEFINITIONS(-Wall -O2 -static -std=c++0x -Wunused-local-typedefs )

#list all source files here
add_subdirectory(src)
add_subdirectory(test)

#ADD_EXECUTABLE(tercpp main.cpp)

#install(TARGETS tercpp DESTINATION bin)

#need to link to some other libraries ? just add them here
#TARGET_LINK_LIBRARIES(tercpp png jpeg)
 
