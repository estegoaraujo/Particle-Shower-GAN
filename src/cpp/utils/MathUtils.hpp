#pragma once
/**
 * @file utils/MathUtils.hpp
 * @brief Physics-oriented math utilities.
 *
 * All functions are constexpr or inline for zero-overhead use
 * inside the hot simulation loop.
 */

#include <cmath>

namespace psg::math {

/// Electron rest mass [GeV/c^2]
inline constexpr float Me  = 0.000511f;

/// Muon rest mass [GeV/c^2]
inline constexpr float Mmu = 0.105658f;

/// Pion rest mass [GeV/c^2]
inline constexpr float Mpi = 0.139570f;

/// Speed of light (natural units: c=1)
inline constexpr float C   = 1.0f;

/// pi
inline constexpr float Pi  = 3.14159265358979323846f;

// ============================================================
//  Relativistic kinematics
// ============================================================

/// Total energy from momentum and mass: E = sqrt(p^2 + m^2)
[[nodiscard]] inline float energyFromMomentum(float p, float m) noexcept
{ return std::sqrt(p*p + m*m); }

/// Momentum from energy and mass: p = sqrt(E^2 - m^2)
[[nodiscard]] inline float momentumFromEnergy(float E, float m) noexcept
{ return (E > m) ? std::sqrt(E*E - m*m) : 0.0f; }

/// Lorentz beta = p/E
[[nodiscard]] inline float beta(float p, float E) noexcept
{ return (E > 0.0f) ? p/E : 0.0f; }

// ============================================================
//  Shower profile functions
// ============================================================

/// Shower maximum depth: t_max = ln(E0/Ec) - 1  (electrons)
[[nodiscard]] inline float showerMaximum(float E0, float Ec,
                                          bool isPhoton = false) noexcept
{
    return std::log(E0 / Ec) - (isPhoton ? 0.5f : 1.0f);
}

/**
 * Longitudinal shower profile (Longo-Sestili / Gamma distribution).
 *
 *   f(t) = b * (b*t)^(a-1) * exp(-b*t) / Gamma(a)
 *
 * where a = tMax*b + 1,  b ~ 0.5 for electrons.
 * Returns fractional energy density at depth t [X0 units].
 */
[[nodiscard]] inline float longitudinalProfile(float t, float tMax,
                                                float b = 0.5f) noexcept
{
    const float a  = tMax * b + 1.0f;
    const float bt = b * t;
    if (bt <= 0.0f) return 0.0f;
    return b * std::pow(bt, a - 1.0f) * std::exp(-bt) / std::tgamma(a);
}

// ============================================================
//  General helpers
// ============================================================

/// Clamp x to [lo, hi]
[[nodiscard]] inline float clamp(float x, float lo, float hi) noexcept
{ return (x < lo) ? lo : (x > hi) ? hi : x; }

/// Linear interpolation
[[nodiscard]] inline float lerp(float a, float b, float t) noexcept
{ return a + t*(b-a); }

} // namespace psg::math
