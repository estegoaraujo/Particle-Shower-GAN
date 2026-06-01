#pragma once

#include <glad/glad.h>
#include "ShaderProgram.hpp"
#include "../core/Particle.hpp"

#include <array>
#include <memory>
#include <vector>

namespace psg {

struct Camera
{
    float azimuth   =  1.0f;
    float elevation =  0.18f;
    float distance  =  350.0f;
    float fovY      =  45.0f;
    float nearPlane =   1.0f;
    float farPlane  = 2000.0f;

    [[nodiscard]] std::array<float, 16> mvp(float aspectRatio) const noexcept;
};

class Renderer
{
public:
    Renderer(const char* vertPath, const char* fragPath);
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    void uploadTracks(const std::vector<Particle>& particles);
    void uploadGanTracks(const std::vector<Particle>& particles);

    void drawSideBySide(int windowWidth, int windowHeight) const;
    void draw(float aspectRatio) const;

    Camera& camera() noexcept { return camera_; }

private:
    struct TrackRange
    {
        GLint   first = 0;
        GLsizei count = 0;
    };

    void uploadToBuffers(const std::vector<Particle>& particles,
                         GLuint vao, GLuint vbo,
                         std::vector<TrackRange>& ranges,
                         float& maxEnergy);

    void drawTracks(const std::vector<TrackRange>& tracks,
                    float maxEnergy, float aspectRatio) const;

    GLuint vao_     = 0;
    GLuint vbo_     = 0;
    GLuint vaoGan_  = 0;
    GLuint vboGan_  = 0;

    std::vector<TrackRange> tracks_;
    std::vector<TrackRange> ganTracks_;

    std::unique_ptr<ShaderProgram> shader_;
    Camera camera_;
    float  maxEnergy_    = 1.0f;
    float  maxEnergyGan_ = 1.0f;

    void initBuffers(GLuint& vao, GLuint& vbo);
};

}

