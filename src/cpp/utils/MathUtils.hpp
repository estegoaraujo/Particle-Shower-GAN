#pragma once
#include <cmath>

namespace psg::math {


inline constexpr float Me  = 0.000511f;

inline constexpr float Mmu = 0.105658f;


inline constexpr float Mpi = 0.139570f;


inline constexpr float C   = 1.0f;


inline constexpr float Pi  = 3.14159265358979323846f;


[[nodiscard]] inline float energyFromMomentum(float p, float m) noexcept
{ return std::sqrt(p*p + m*m); }


[[nodiscard]] inline float momentumFromEnergy(float E, float m) noexcept
{ return (E > m) ? std::sqrt(E*E - m*m) : 0.0f; }

[[nodiscard]] inline float beta(float p, float E) noexcept
{ return (E > 0.0f) ? p/E : 0.0f; }

[[nodiscard]] inline float showerMaximum(float E0, float Ec,
                                          bool isPhoton = false) noexcept
{
    return std::log(E0 / Ec) - (isPhoton ? 0.5f : 1.0f);
}


[[nodiscard]] inline float longitudinalProfile(float t, float tMax,
                                                float b = 0.5f) noexcept
{
    const float a  = tMax * b + 1.0f;
    const float bt = b * t;
    if (bt <= 0.0f) return 0.0f;
    return b * std::pow(bt, a - 1.0f) * std::exp(-bt) / std::tgamma(a);
}


[[nodiscard]] inline float clamp(float x, float lo, float hi) noexcept
{ return (x < lo) ? lo : (x > hi) ? hi : x; }

[[nodiscard]] inline float lerp(float a, float b, float t) noexcept
{ return a + t*(b-a); }

}
