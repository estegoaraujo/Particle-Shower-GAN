#pragma once
/**
 * @file core/Particle.hpp
 * @brief Defines the Particle state vector and type taxonomy.
 *
 * Physics Design Notes
 * ====================
 * A relativistic particle is fully described at any instant by:
 *
 *   State S = { position x, momentum p, energy E, type, status }
 *
 * Energy-momentum relation (special relativity):
 *   E^2 = (pc)^2 + (m0*c^2)^2
 *
 * At each simulation step ds the particle propagates and loses energy:
 *   E(s + ds) = E(s) - (dE/dx)*ds  - dE_stoch
 *
 * where (dE/dx) is the continuous Bethe-Bloch ionisation loss and
 * dE_stoch is the discrete stochastic term (bremsstrahlung / pair production).
 */

#include <array>
#include <cstdint>
#include <vector>

namespace psg {

// ============================================================
//  Particle Type Taxonomy
// ============================================================

/**
 * @enum ParticleType
 * PDG codes:  Photon=22, Electron=11, Muon=13, PiPlus=211
 */
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
    Active   = 0,   ///< Propagating through detector
    Stopped  = 1,   ///< Energy fell below threshold E_cut
    Escaped  = 2,   ///< Left the DetectorVolume boundary
    Decayed  = 3    ///< Underwent discrete process (branched into daughters)
};

// ============================================================
//  3-Vector
// ============================================================
struct Vec3
{
    float x = 0.0f, y = 0.0f, z = 0.0f;

    [[nodiscard]] float dot(const Vec3& o)  const noexcept { return x*o.x + y*o.y + z*o.z; }
    [[nodiscard]] float norm2()             const noexcept { return dot(*this); }
    [[nodiscard]] Vec3  normalised()        const noexcept;

    Vec3 operator+(const Vec3& o) const noexcept { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator*(float s)       const noexcept { return {x*s,   y*s,   z*s};   }
};

// ============================================================
//  Particle State Vector
// ============================================================
/**
 * Export format (flat float32, 10 elements):
 *   [ pos.x, pos.y, pos.z, dir.x, dir.y, dir.z,
 *     energy, totalPathLength, type_f32, status_f32 ]
 */
struct Particle
{
    Vec3  position  = {};
    Vec3  direction = {0.0f, 0.0f, 1.0f};
    float energy    = 0.0f;                 ///< Total energy [GeV]

    ParticleType   type     = ParticleType::Electron;
    ParticleStatus status   = ParticleStatus::Active;
    int32_t        id       = -1;
    int32_t        parentId = -1;

    float totalPathLength    = 0.0f;  ///< Cumulative path [cm]
    float totalEnergyDeposit = 0.0f;  ///< Cumulative dE  [GeV]

    std::vector<Vec3> trackPoints;    ///< Sampled positions for renderer

    // --------------------------------------------------------
    //  Physics helpers (implemented in Particle.cpp, Phase 5)
    // --------------------------------------------------------

    /// Rest mass m0 [GeV/c^2]  (PDG 2022 values)
    [[nodiscard]] float restMass() const noexcept;

    /// Momentum |p|c = sqrt(E^2 - m0^2) [GeV/c]
    [[nodiscard]] float momentum() const noexcept;

    /// Lorentz gamma = E / m0  (undefined for photons)
    [[nodiscard]] float lorentzGamma() const noexcept;

    /// Serialise to flat float array for Python/PyTorch export
    [[nodiscard]] std::array<float, 10> toFloatArray() const noexcept;
};

} // namespace psg
