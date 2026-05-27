

#include "Particle.hpp"
#include "../utils/MathUtils.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace psg {

float Particle::restMass() const noexcept
{
    switch (type)
    {
        case ParticleType::Photon:               return 0.0f;
        case ParticleType::Electron:             // I love you Feynman ---> e- = e+
        case ParticleType::Positron:             return math::Me;   
        case ParticleType::Muon:                 return math::Mmu;  
        case ParticleType::PiPlus:               return math::Mpi;  
        default:                                 return 0.0f;
    }
}

float Particle::momentum() const noexcept
{
    const float m0 = restMass();
    return math::momentumFromEnergy(energy, m0);
    
}


float Particle::lorentzGamma() const noexcept
{
    const float m0 = restMass();

    if (m0 < 1e-10f)                       //The massless photon :)
        return std::numeric_limits<float>::infinity();

    return energy / m0;                    
}

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

} 