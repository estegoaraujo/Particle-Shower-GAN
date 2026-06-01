#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "core/Particle.hpp"
#include "core/DetectorVolume.hpp"
#include "core/ShowerSimulation.hpp"
#include "core/Exporter.hpp"
#include "core/NpyLoader.hpp"
#include "renderer/Renderer.hpp"
#include "utils/Logger.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

static constexpr int   WIN_WIDTH  = 1280;
static constexpr int   WIN_HEIGHT = 720;
static constexpr char  WIN_TITLE[]= "Particle-Shower-GAN  |  LEFT: Real  |  RIGHT: GAN";

static constexpr float BG_R = 0x0A / 255.0f;
static constexpr float BG_G = 0x0A / 255.0f;
static constexpr float BG_B = 0x0A / 255.0f;

static psg::Renderer* g_renderer  = nullptr;
static int   g_fbW      = WIN_WIDTH;
static int   g_fbH      = WIN_HEIGHT;
static bool  g_mouseDown = false;
static float g_lastMouseX = 0.0f;
static float g_lastMouseY = 0.0f;
static bool  g_autoRotate = true;

static void framebufferSizeCallback(GLFWwindow*, int w, int h)
{
    g_fbW = w; g_fbH = h;
}

static void keyCallback(GLFWwindow* window, int key, int, int action, int)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        g_autoRotate = !g_autoRotate;
}

static void scrollCallback(GLFWwindow*, double, double yoffset)
{
    if (g_renderer)
    {
        g_renderer->camera().distance -= static_cast<float>(yoffset) * 10.0f;
        g_renderer->camera().distance  =
            std::max(50.0f, g_renderer->camera().distance);
    }
}

static void mouseButtonCallback(GLFWwindow*, int button, int action, int)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)  { g_mouseDown = true;  g_autoRotate = false; }
        if (action == GLFW_RELEASE){ g_mouseDown = false; }
    }
}

static void cursorPosCallback(GLFWwindow*, double xpos, double ypos)
{
    const float x = static_cast<float>(xpos);
    const float y = static_cast<float>(ypos);

    if (g_mouseDown && g_renderer)
    {
        const float sensitivity = 0.005f;
        g_renderer->camera().azimuth   += (x - g_lastMouseX) * sensitivity;
        g_renderer->camera().elevation -= (y - g_lastMouseY) * sensitivity;
        g_renderer->camera().elevation  =
            std::max(-1.5f, std::min(1.5f, g_renderer->camera().elevation));
    }
    g_lastMouseX = x;
    g_lastMouseY = y;
}

[[nodiscard]] static int fatalError(const char* msg)
{
    std::cerr << "[FATAL] " << msg << '\n';
    glfwTerminate();
    return EXIT_FAILURE;
}

static int runGenerate(int n)
{
    PSG_LOG_INFO("Generate mode:", n, "showers → data/raw/");
    std::filesystem::create_directories("data/raw");
    psg::Exporter::generateDataset(
        "data/raw", n,
        {10.0f, 50.0f, 100.0f, 200.0f, 500.0f},
        psg::ParticleType::Electron, 1);
    return EXIT_SUCCESS;
}

static int runVisualise(const std::string& ganFile)
{
    if (!glfwInit())
        return fatalError("glfwInit() failed.");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT,
                                          WIN_TITLE, nullptr, nullptr);
    if (!window)
        return fatalError("glfwCreateWindow() failed.");

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        return fatalError("gladLoadGLLoader() failed.");

    PSG_LOG_INFO("OpenGL", glGetString(GL_VERSION),
                 "| Renderer:", glGetString(GL_RENDERER));

    {
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        g_fbW = fbW; g_fbH = fbH;
    }

    psg::SimConfig config;
    config.stepSize     = 0.5f;
    config.eCut         = 0.001f;
    config.maxParticles = 20000;
    config.seed         = 42;

    psg::DetectorVolume detector;
    psg::ShowerSimulation sim(config, detector);
    sim.seedPrimary(psg::ParticleType::Electron, 100.0f,
                    {0.0f, 0.0f, -170.0f},
                    {0.0f, 0.0f,  1.0f});

    PSG_LOG_INFO("Running shower simulation...");
    sim.run();
    PSG_LOG_INFO("Done.", sim.totalCount(), "particles.");

    psg::Renderer renderer(
        "shaders/particle_track.vert",
        "shaders/particle_track.frag"
    );
    g_renderer = &renderer;
    renderer.uploadTracks(sim.allParticles());

    if (!ganFile.empty())
    {
        try
        {
            PSG_LOG_INFO("Loading GAN file:", ganFile);
            psg::VoxelGrid grid = psg::NpyLoader::load(ganFile);
            auto ganParticles   = psg::NpyLoader::voxelsToParticles(
                                      grid, detector);
            renderer.uploadGanTracks(ganParticles);
            PSG_LOG_INFO("GAN shower loaded:", ganParticles.size(), "voxel segments.");
        }
        catch (const std::exception& e)
        {
            PSG_LOG_WARN("Could not load GAN file:", e.what());
            PSG_LOG_WARN("Showing only real shower.");
        }
    }

    glClearColor(BG_R, BG_G, BG_B, 1.0f);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float azimuthOffset = 0.0f;
    const bool sideBySide = !ganFile.empty();

    PSG_LOG_INFO("Controls — drag: rotate | scroll: zoom | space: auto-rotate | ESC: exit");
    if (sideBySide)
        PSG_LOG_INFO("Side-by-side mode: LEFT = Real C++  |  RIGHT = GAN");

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        if (g_autoRotate)
        {
            azimuthOffset += 0.002f;
            renderer.camera().azimuth   = 1.0f + azimuthOffset;
            renderer.camera().elevation = 0.18f;
        }

        glClear(GL_COLOR_BUFFER_BIT);

        if (sideBySide)
            renderer.drawSideBySide(g_fbW, g_fbH);
        else
            renderer.draw(static_cast<float>(g_fbW) /
                          static_cast<float>(g_fbH));

        glfwSwapBuffers(window);
    }

    g_renderer = nullptr;
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    PSG_LOG_INFO("Particle-Shower-GAN v0.1");

    if (argc >= 3 && std::strcmp(argv[1], "--generate") == 0)
    {
        const int n = std::atoi(argv[2]);
        if (n <= 0) { std::cerr << "Usage: psg --generate <N>\n"; return EXIT_FAILURE; }
        return runGenerate(n);
    }

    std::string ganFile = "";
    if (argc >= 3 && std::strcmp(argv[1], "--gan") == 0)
        ganFile = argv[2];

    return runVisualise(ganFile);
}
