cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

set(CMAKE_PREFIX_PATH "/home/wabby/Project/FlashAttention/libtorch/share/cmake/Torch;${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# libtorch的测试
add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# FlashAttention（ours）
add_executable(flash-atten-main flash-atten-main.cpp flash_learn.cu)
set_property(TARGET flash-atten-main PROPERTY CXX_STANDARD 17)
find_package(CUDA REQUIRED)
target_link_libraries(flash-atten-main
    PRIVATE
        ${TORCH_LIBRARIES}
        CUDA::curand
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(flash-atten-main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# FlashAttention（torch）
add_executable(attn-app test-attention.cpp)
set_property(TARGET attn-app PROPERTY CXX_STANDARD 17)
find_package(CUDA REQUIRED)
target_link_libraries(attn-app
    PRIVATE
        ${TORCH_LIBRARIES}
        CUDA::curand
)

# cuRAND的测试
add_executable(mt19937-app curand_mt19937_lognormal_example.cpp)
set_property(TARGET mt19937-app PROPERTY CXX_STANDARD 17)
find_package(CUDA REQUIRED)
target_link_libraries(mt19937-app
    PRIVATE
        ${TORCH_LIBRARIES}
        CUDA::curand
)