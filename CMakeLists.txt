cmake_minimum_required(VERSION 3.20)
project(ParticleShowerGAN VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3 -Wall -Wextra -Wpedantic")

# ── Dependencies ──────────────────────────────────────────────
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

# GLAD (vendored)
add_library(glad STATIC extern/glad/src/glad.c)
target_include_directories(glad PUBLIC extern/glad/include)

# ── Sources ───────────────────────────────────────────────────
set(SOURCES
    src/cpp/main.cpp
    src/cpp/core/particle.cpp
    src/cpp/core/ShowerSimulation.cpp
    src/cpp/renderer/renderer.cpp
    src/cpp/renderer/ShaderProgram.cpp
)

add_executable(psg ${SOURCES})

target_include_directories(psg PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/extern/glad/include
)

target_link_libraries(psg PRIVATE
    glad
    glfw
    OpenGL::GL
    $<$<PLATFORM_ID:Linux>:dl>
    $<$<PLATFORM_ID:Linux>:pthread>
)

target_compile_features(psg PRIVATE cxx_std_20)
