/**
 * @file core/Particle.cpp
 * @brief Implements Particle kinematics and Vec3 geometry.
 *
 * All physics uses natural units: c = 1.
 * Energies and masses in GeV, distances in cm.
 */

#include "Particle.hpp"
#include "../utils/MathUtils.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace psg {

// ============================================================
//  Vec3
// ============================================================

/**
 * Returns a unit vector (length = 1) in the same direction.
 *
 * Why we need this:
 *   Direction vectors can drift away from unit length due to
 *   floating-point accumulation errors over many steps.
 *   Renormalising keeps the physics correct.
 *
 * Formula:  v_hat = v / |v|
 *           where |v| = sqrt(x^2 + y^2 + z^2)
 */
Vec3 Vec3::normalised() const noexcept
{
    const float len2 = norm2();         // |v|^2 = x^2 + y^2 + z^2

    // Guard: if the vector is essentially zero, return +z axis.
    // This can happen for a particle created at rest — it has no
    // meaningful direction yet.
    if (len2 < 1e-12f)
        return {0.0f, 0.0f, 1.0f};

    const float invLen = 1.0f / std::sqrt(len2);  // 1 / |v|
    return { x * invLen, y * invLen, z * invLen };
}

// ============================================================
//  Particle — rest mass
// ============================================================

/**
 * Returns the rest mass m0 [GeV/c^2] for this particle type.
 *
 * These are fundamental physical constants — they never change.
 * Values from the Particle Data Group (PDG) Review 2022.
 *
 * The photon is special: it is massless. It still carries energy
 * and momentum via E = |p|c (or E = |p| in natural units),
 * but has no rest frame.
 */
float Particle::restMass() const noexcept
{
    switch (type)
    {
        case ParticleType::Photon:               return 0.0f;
        case ParticleType::Electron:             // fall-through: e- = e+
        case ParticleType::Positron:             return math::Me;   // 0.000511 GeV
        case ParticleType::Muon:                 return math::Mmu;  // 0.105658 GeV
        case ParticleType::PiPlus:               return math::Mpi;  // 0.139570 GeV
        default:                                 return 0.0f;
    }
}

// ============================================================
//  Particle — momentum magnitude
// ============================================================

/**
 * Returns the momentum magnitude |p| [GeV/c].
 *
 * Derived from Einstein's energy-momentum relation:
 *
 *   E^2 = p^2 + m0^2        (natural units, c=1)
 *
 * Solving for p:
 *
 *   p = sqrt(E^2 - m0^2)
 *
 * Physical interpretation:
 *   - For a photon (m0=0):    p = E        (all energy is kinetic)
 *   - For a slow particle:    p ≈ m0*v     (classical limit)
 *   - For ultra-relativistic: p ≈ E        (mass negligible)
 *
 * We clamp to 0 if E < m0 (unphysical — guards against
 * floating-point drift pushing energy just below rest mass).
 */
float Particle::momentum() const noexcept
{
    const float m0 = restMass();
    return math::momentumFromEnergy(energy, m0);
    // = (energy > m0) ? sqrt(energy*energy - m0*m0) : 0.0f
}

// ============================================================
//  Particle — Lorentz factor
// ============================================================

/**
 * Returns the Lorentz factor gamma = E / (m0 * c^2).
 *
 * Physical meaning:
 *   gamma tells us how much the particle's relativistic mass
 *   exceeds its rest mass, and how much time dilation it experiences.
 *
 *   gamma = 1          → at rest
 *   gamma = 1.00001    → everyday speeds (car, plane)
 *   gamma = 10         → cosmic ray muon reaching Earth's surface
 *   gamma = 200,000    → LHC proton at 7 TeV
 *
 * For photons: m0 = 0, so gamma is mathematically undefined
 * (infinite). We return std::numeric_limits<float>::infinity()
 * to signal this to the caller.
 *
 * Note: the simulation does NOT use gamma directly in the energy
 * loss model — it appears here for diagnostic/export purposes.
 */
float Particle::lorentzGamma() const noexcept
{
    const float m0 = restMass();

    if (m0 < 1e-10f)                        // photon (massless)
        return std::numeric_limits<float>::infinity();

    return energy / m0;                     // gamma = E / m0  (natural units)
}

// ============================================================
//  Particle — serialise to float array (for Python export)
// ============================================================

/**
 * Packs the particle state into 10 contiguous float32 values.
 *
 * Layout (matches the Python loader in src/python/data/loader.py):
 *
 *   Index  Field              Units
 *   -----  -----------------  ------
 *     0    position.x         cm
 *     1    position.y         cm
 *     2    position.z         cm
 *     3    direction.x        (unitless — unit vector component)
 *     4    direction.y
 *     5    direction.z
 *     6    energy             GeV
 *     7    totalPathLength    cm
 *     8    type               (cast to float: 0=photon,1=e-,2=e+,...)
 *     9    status             (cast to float: 0=active,1=stopped,...)
 *
 * Why float32?
 *   PyTorch's default tensor dtype is float32. Using the same type
 *   in C++ means we can memcpy the raw bytes directly into a
 *   torch::Tensor without any conversion — zero overhead.
 *
 * The Python side does:
 *   data = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, 10)
 *   tensor = torch.from_numpy(data)
 */
std::array<float, 10> Particle::toFloatArray() const noexcept
{
    return {
        position.x,
        position.y,
        position.z,
        direction.x,
        direction.y,
        direction.z,
        energy,
        totalPathLength,
        static_cast<float>(type),
        static_cast<float>(status)
    };
}

} // namespace psg
