
#include "Renderer.hpp"
#include "../utils/Logger.hpp"

#include <cmath>
#include <stdexcept>

namespace psg {

std::array<float, 16> Camera::mvp(float aspect) const noexcept
{

    const float cosEl = std::cos(elevation);
    const float sinEl = std::sin(elevation);
    const float cosAz = std::cos(azimuth);
    const float sinAz = std::sin(azimuth);

    const float ex = distance * cosEl * sinAz;
    const float ey = distance * sinEl;
    const float ez = distance * cosEl * cosAz;

  
    const float fLen = std::sqrt(ex*ex + ey*ey + ez*ez);
    const float fx = -ex/fLen, fy = -ey/fLen, fz = -ez/fLen;

  
    const float ux = 0.0f, uy = 1.0f, uz = 0.0f;


    float rx = fy*uz - fz*uy;
    float ry = fz*ux - fx*uz;
    float rz = fx*uy - fy*ux;
    const float rLen = std::sqrt(rx*rx + ry*ry + rz*rz);
    rx /= rLen; ry /= rLen; rz /= rLen;


    const float cx = ry*fz - rz*fy;
    const float cy = rz*fx - rx*fz;
    const float cz = rx*fy - ry*fx;


    const float tdx = -(rx*ex + ry*ey + rz*ez);
    const float tdy = -(cx*ex + cy*ey + cz*ez);
    const float tdz =  (fx*ex + fy*ey + fz*ez);


    const std::array<float,16> V = {
        rx,  ry,  rz,  0.0f,  
        cx,  cy,  cz,  0.0f,   
       -fx, -fy, -fz,  0.0f,
       tdx, tdy, tdz,  1.0f    
    };

    
    const float fovRad = fovY * (3.14159265f / 180.0f);
    const float f      = 1.0f / std::tan(fovRad * 0.5f);
    const float nf     = nearPlane - farPlane;

  
    const std::array<float,16> P = {
        f/aspect,  0.0f,  0.0f,                          0.0f,  // col 0
        0.0f,      f,     0.0f,                          0.0f,  // col 1
        0.0f,      0.0f,  (farPlane+nearPlane)/nf,      -1.0f,  // col 2
        0.0f,      0.0f,  (2.0f*farPlane*nearPlane)/nf,  0.0f   // col 3
    };


    std::array<float,16> MVP{};
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
                sum += P[k*4 + row] * V[col*4 + k];
            MVP[col*4 + row] = sum;
        }
    return MVP;
}


Renderer::Renderer(const char* vertPath, const char* fragPath)
{
    shader_ = std::make_unique<ShaderProgram>(vertPath, fragPath);

  
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);

  
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);

  
    const GLsizei stride = 4 * sizeof(float);


    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          stride, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);


    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE,
                          stride, reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    PSG_LOG_INFO("Renderer initialised. VAO:", vao_, "VBO:", vbo_);
}


Renderer::~Renderer()
{
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (vao_) glDeleteVertexArrays(1, &vao_);
}


void Renderer::uploadTracks(const std::vector<Particle>& particles)
{
    tracks_.clear();
    std::vector<float> buffer;
    buffer.reserve(particles.size() * 256 * 4);  

    maxEnergy_ = 0.001f;  

    for (const auto& p : particles)
    {
    
        if (p.trackPoints.size() < 2) continue;

    
        const GLint firstVertex = static_cast<GLint>(buffer.size() / 4);
        const GLsizei count     = static_cast<GLsizei>(p.trackPoints.size());

   
        const float trackEnergy = p.trackPoints.empty() ? 0.0f
                                  : p.energy + p.totalEnergyDeposit;
        maxEnergy_ = std::max(maxEnergy_, trackEnergy);

      
        for (GLsizei i = 0; i < count; ++i)
        {
            const Vec3& pt = p.trackPoints[static_cast<size_t>(i)];
            const float t  = static_cast<float>(i) / static_cast<float>(count - 1);
            const float e  = trackEnergy * (1.0f - t); 

            buffer.push_back(pt.x);
            buffer.push_back(pt.y);
            buffer.push_back(pt.z);
            buffer.push_back(e);
        }

        tracks_.push_back({ firstVertex, count });
    }

    if (buffer.empty())
    {
        PSG_LOG_WARN("uploadTracks: no track data to upload.");
        return;
    }

  
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(buffer.size() * sizeof(float)),
                 buffer.data(),
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    PSG_LOG_INFO("Uploaded", tracks_.size(), "tracks,",
                 buffer.size()/4, "vertices to GPU.");
}


void Renderer::draw(float aspectRatio) const
{
    if (tracks_.empty()) return;

    shader_->use();

    shader_->setMat4("uMVP", camera_.mvp(aspectRatio));

    shader_->setFloat("uMaxEnergy", maxEnergy_);

    glBindVertexArray(vao_);

  
    for (const auto& track : tracks_)
    {
    
        glDrawArrays(GL_LINE_STRIP, track.first, track.count);
    }

    glBindVertexArray(0);
}

} 
