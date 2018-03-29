PROJECT(checkbitexts)


include_directories(BEFORE ${CMAKE_SOURCE_DIR}/src)

SET(tercpp_SRCS
   main.cpp
   ${CMAKE_SOURCE_DIR}/src/bitexts.cpp
)

add_executable(scoreBitext ${tercpp_SRCS})
target_link_libraries(scoreBitext multivec)

install(TARGETS scoreBitext DESTINATION bin)

