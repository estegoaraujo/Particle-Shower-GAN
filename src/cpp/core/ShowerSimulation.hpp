#pragma once

#include "Particle.hpp"
#include "DetectorVolume.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace psg {

struct SimConfig
{
    float    stepSize     = 0.5f;
    float    eCut         = 0.001f;
    float    splitFrac    = 0.5f;
    int      maxParticles = 50000;
    int      maxSteps     = 100000;
    uint32_t seed         = 42;
};

class ShowerSimulation
{
public:
    ShowerSimulation(const SimConfig& config,
                     DetectorVolume&  detector);

    ShowerSimulation(const ShowerSimulation&)            = delete;
    ShowerSimulation& operator=(const ShowerSimulation&) = delete;

    void seedPrimary(ParticleType type,
                     float        energy,
                     Vec3         pos = {0.0f, 0.0f, 0.0f},
                     Vec3         dir = {0.0f, 0.0f, 1.0f});

    void run();
    void step();

    [[nodiscard]] bool  isFinished()     const noexcept;
    [[nodiscard]] int   activeCount()    const noexcept;
    [[nodiscard]] int   totalCount()     const noexcept { return static_cast<int>(particles_.size()); }
    [[nodiscard]] float primaryEnergy()  const noexcept { return primaryEnergy_; }

    [[nodiscard]] const std::vector<Particle>& allParticles() const noexcept
    { return particles_; }

    [[nodiscard]] float computeDEdx(const Particle& p) const noexcept;
    [[nodiscard]] bool  sampleBrem(const Particle& p);
    [[nodiscard]] bool  samplePair(const Particle& p);

private:
    void stepParticle(Particle& p);
    void branchFrom(Particle& parent, ParticleType d1, ParticleType d2);
    int  nextId() noexcept { return nextId_++; }

    SimConfig             config_;
    DetectorVolume&       detector_;
    std::vector<Particle> particles_;
    std::mt19937          rng_;
    std::uniform_real_distribution<float> uniform01_{0.0f, 1.0f};
    float                 primaryEnergy_ = 0.0f;
    int                   nextId_        = 0;
    int                   stepCount_     = 0;
};

}
