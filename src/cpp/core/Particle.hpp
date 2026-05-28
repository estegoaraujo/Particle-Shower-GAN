#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace psg {

enum class ParticleType : uint8_t
{
    Photon   = 0,
    Electron = 1,
    Positron = 2,
    Muon     = 3,
    PiPlus   = 4,
    Unknown  = 255
};

enum class ParticleStatus : uint8_t
{
    Active   = 0,
    Stopped  = 1,
    Escaped  = 2,
    Decayed  = 3
};

struct Vec3
{
    float x = 0.0f, y = 0.0f, z = 0.0f;

    [[nodiscard]] float dot(const Vec3& o)  const noexcept { return x*o.x + y*o.y + z*o.z; }
    [[nodiscard]] float norm2()             const noexcept { return dot(*this); }
    [[nodiscard]] Vec3 normalised() const noexcept {
        const float n = std::sqrt(norm2());
        return (n > 0.0f) ? Vec3{x/n, y/n, z/n} : Vec3{0.0f, 0.0f, 1.0f};
    }

    Vec3 operator+(const Vec3& o) const noexcept { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator*(float s)       const noexcept { return {x*s,   y*s,   z*s};   }
};

struct Particle
{
    Vec3  position  = {};
    Vec3  direction = {0.0f, 0.0f, 1.0f};
    float energy    = 0.0f;

    ParticleType   type     = ParticleType::Electron;
    ParticleStatus status   = ParticleStatus::Active;
    int32_t        id       = -1;
    int32_t        parentId = -1;

    float totalPathLength    = 0.0f;
    float totalEnergyDeposit = 0.0f;

    std::vector<Vec3> trackPoints;

    [[nodiscard]] float restMass()     const noexcept;
    [[nodiscard]] float momentum()     const noexcept;
    [[nodiscard]] float lorentzGamma() const noexcept;
    [[nodiscard]] std::array<float, 10> toFloatArray() const noexcept;
};

}
