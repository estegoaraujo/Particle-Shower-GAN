
#include <glad/glad.h>

#include <string>
#include <string_view>
#include <unordered_map>
#include <array>

namespace psg {

class ShaderProgram
{
public:
  
    ShaderProgram(std::string_view vertPath, std::string_view fragPath);

    ~ShaderProgram();

  
    ShaderProgram(const ShaderProgram&)            = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;
    ShaderProgram(ShaderProgram&& o) noexcept;

    void use() const noexcept;

    void setFloat(std::string_view name, float value);
    void setMat4 (std::string_view name, const std::array<float,16>& m);
    void setVec3 (std::string_view name, float x, float y, float z);

    [[nodiscard]] GLuint id() const noexcept { return programId_; }

private:
    GLuint programId_ = 0;
    mutable std::unordered_map<std::string, GLint> uniformCache_;

    static GLuint  compileShader(GLenum type, const std::string& source);
    static std::string loadFile(std::string_view path);
    GLint          uniformLoc(std::string_view name) const;
};

} 
