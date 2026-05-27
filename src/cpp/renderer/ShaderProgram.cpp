
#include "ShaderProgram.hpp"
#include "../utils/Logger.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace psg {

ShaderProgram::ShaderProgram(std::string_view vertPath,
                             std::string_view fragPath)
{
  
    const std::string vertSrc = loadFile(vertPath);
    const std::string fragSrc = loadFile(fragPath);

    const GLuint vert = compileShader(GL_VERTEX_SHADER,   vertSrc);
    const GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);

 
    programId_ = glCreateProgram();
    glAttachShader(programId_, vert);
    glAttachShader(programId_, frag);
    glLinkProgram(programId_);

 
    GLint success = 0;
    glGetProgramiv(programId_, GL_LINK_STATUS, &success);
    if (!success)
    {
        char log[1024];
        glGetProgramInfoLog(programId_, sizeof(log), nullptr, log);
        glDeleteProgram(programId_);
        programId_ = 0;
        throw std::runtime_error(
            std::string("Shader link failed:\n") + log);
    }

    
    glDeleteShader(vert);
    glDeleteShader(frag);

    PSG_LOG_INFO("ShaderProgram linked. GL id:", programId_);
}

ShaderProgram::~ShaderProgram()
{
    if (programId_)
        glDeleteProgram(programId_);
}

ShaderProgram::ShaderProgram(ShaderProgram&& o) noexcept
    : programId_(o.programId_)
    , uniformCache_(std::move(o.uniformCache_))
{
    o.programId_ = 0;   
}


void ShaderProgram::use() const noexcept
{
    glUseProgram(programId_);
}


void ShaderProgram::setFloat(std::string_view name, float value)
{
    glUniform1f(uniformLoc(name), value);
}

void ShaderProgram::setMat4(std::string_view name,
                             const std::array<float, 16>& m)
{

    glUniformMatrix4fv(uniformLoc(name), 1, GL_FALSE, m.data());
}

void ShaderProgram::setVec3(std::string_view name,
                             float x, float y, float z)
{
    glUniform3f(uniformLoc(name), x, y, z);
}


GLuint ShaderProgram::compileShader(GLenum type, const std::string& source)
{
    const GLuint id = glCreateShader(type);
    const char*  src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    GLint success = 0;
    glGetShaderiv(id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char log[1024];
        glGetShaderInfoLog(id, sizeof(log), nullptr, log);
        glDeleteShader(id);
        const char* typeName = (type == GL_VERTEX_SHADER)
                               ? "Vertex" : "Fragment";
        throw std::runtime_error(
            std::string(typeName) + " shader compile failed:\n" + log);
    }
    return id;
}

std::string ShaderProgram::loadFile(std::string_view path)
{
    std::ifstream file{std::string(path)};
    if (!file.is_open())
        throw std::runtime_error(
            "ShaderProgram: cannot open file: " + std::string(path));

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}


GLint ShaderProgram::uniformLoc(std::string_view name) const
{
    const std::string key{name};
    auto it = uniformCache_.find(key);
    if (it != uniformCache_.end())
        return it->second;

    const GLint loc = glGetUniformLocation(programId_, key.c_str());
    if (loc == -1)
        PSG_LOG_WARN("Uniform not found in shader:", key);

    uniformCache_[key] = loc;
    return loc;
}

} 
