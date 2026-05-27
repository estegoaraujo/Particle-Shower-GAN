cmake_minimum_required(VERSION 3.20)
project(ParticleShowerGAN LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── GLFW (system package: sudo apt install libglfw3-dev) ───────
find_package(glfw3 3.3 REQUIRED)

# ── GLAD (vendored in extern/) ─────────────────────────────────
add_library(glad STATIC extern/glad/src/glad.c)
target_include_directories(glad PUBLIC extern/glad/include)

# ── OpenGL ─────────────────────────────────────────────────────
find_package(OpenGL REQUIRED)

# ── Main executable ────────────────────────────────────────────
set(SRC src/cpp)

add_executable(psg
    ${SRC}/main.cpp
    ${SRC}/core/ShowerSimulator.cpp
    ${SRC}/core/particle.cpp
    ${SRC}/renderer/renderer.cpp
    ${SRC}/renderer/ShaderProgram.cpp
)

target_include_directories(psg PRIVATE ${SRC})
target_link_libraries(psg PRIVATE glad glfw OpenGL::GL)
