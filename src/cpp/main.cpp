
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "core/Particle.hpp"
#include "core/DetectorVolume.hpp"
#include "core/ShowerSimulator.hpp"
#include "renderer/Renderer.hpp"
#include "utils/Logger.hpp"

#include <cstdlib>
#include <iostream>


static constexpr int   WIN_WIDTH  = 1280;
static constexpr int   WIN_HEIGHT = 720;
static constexpr char  WIN_TITLE[]= "Particle-Shower-GAN";

static constexpr float BG_R = 0x0A / 255.0f;
static constexpr float BG_G = 0x0A / 255.0f;
static constexpr float BG_B = 0x0A / 255.0f;

static psg::Renderer* g_renderer = nullptr;  
static int  g_fbW = WIN_WIDTH;
static int  g_fbH = WIN_HEIGHT;


static void framebufferSizeCallback(GLFWwindow*, int w, int h)
{
    g_fbW = w; g_fbH = h;
    glViewport(0, 0, w, h);
}

static void keyCallback(GLFWwindow* window, int key, int, int action, int)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
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


[[nodiscard]] static int fatalError(const char* msg)
{
    std::cerr << "[FATAL] " << msg << '\n';
    glfwTerminate();
    return EXIT_FAILURE;
}


int main()
{
    PSG_LOG_INFO("Particle-Shower-GAN v0.1 starting...");

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

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        return fatalError("gladLoadGLLoader() failed.");

    PSG_LOG_INFO("OpenGL", glGetString(GL_VERSION),
                 "| Renderer:", glGetString(GL_RENDERER));

    {
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        g_fbW = fbW; g_fbH = fbH;
        glViewport(0, 0, fbW, fbH);
    }

 
    psg::SimConfig config;
    config.stepSize    = 0.5f;
    config.eCut        = 0.001f;
    config.maxParticles = 20000;
    config.seed        = 42;

    psg::DetectorVolume detector;  

    psg::ShowerSimulation sim(config, detector);
    sim.seedPrimary(psg::ParticleType::Electron,
                    100.0f,        
                    {0.0f, 0.0f, -170.0f}, 
                    {0.0f, 0.0f,  1.0f});   

    PSG_LOG_INFO("Running shower simulation...");
    sim.run();
    PSG_LOG_INFO("Simulation complete.", sim.totalCount(), "particles.");

 
    psg::Renderer renderer(
        "shaders/particle_track.vert",
        "shaders/particle_track.frag"
    );
    g_renderer = &renderer;

    renderer.uploadTracks(sim.allParticles());

  
    PSG_LOG_INFO("Entering render loop. Scroll to zoom, ESC to exit.");

    glClearColor(BG_R, BG_G, BG_B, 1.0f);


    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  
    float azimuthOffset = 0.0f;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        azimuthOffset += 0.002f;
        renderer.camera().azimuth   = 1.0f + azimuthOffset;
        renderer.camera().elevation = 0.18f;

        glClear(GL_COLOR_BUFFER_BIT);

        const float aspect = static_cast<float>(g_fbW) /
                             static_cast<float>(g_fbH);
        renderer.draw(aspect);

        glfwSwapBuffers(window);
    }

    g_renderer = nullptr;
    PSG_LOG_INFO("Shutting down.");
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
