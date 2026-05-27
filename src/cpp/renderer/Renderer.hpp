
#include <glad/glad.h>
#include "ShaderProgram.hpp"
#include "../core/Particle.hpp"

#include <array>
#include <vector>
#include <memory>

namespace psg {

struct Camera
{
    float azimuth   =  0.5f;   
    float elevation =  0.4f;    
    float distance  = 400.0f;  
    float fovY      = 45.0f;    
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


    void draw(float aspectRatio) const;

    Camera& camera() noexcept { return camera_; }

private:

    GLuint vao_      = 0;
    GLuint vbo_      = 0;


    struct TrackRange
    {
        GLint  first = 0;   
        GLsizei count = 0;  
    };
    std::vector<TrackRange> tracks_;
    std::unique_ptr<ShaderProgram> shader_;
    Camera                         camera_;
    float                          maxEnergy_ = 1.0f;
};

} 
