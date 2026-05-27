
#include "ShowerSimulator.hpp"
#include "../utils/MathUtils.hpp"
#include "../utils/Logger.hpp"

#include <cmath>
#include <stdexcept>

namespace psg {

ShowerSimulation::ShowerSimulation(const SimConfig&      config,
                                   const DetectorVolume& detector)
    : config_(config)
    , detector_(detector)
{
    if (config_.seed == 0)
        rng_.seed(std::random_device{}());
    else
        rng_.seed(config_.seed);

   
    particles_.reserve(4096);

    PSG_LOG_INFO("ShowerSimulation created. Material:", detector_.material.name,
                 "| X0:", detector_.material.radiationLength, "cm",
                 "| Ec:", detector_.material.criticalEnergy, "GeV");
}

void ShowerSimulation::seedPrimary(ParticleType type,
                                   float        energy,
                                   Vec3         pos,
                                   Vec3         dir)
{
    if (energy <= config_.eCut)
        throw std::invalid_argument("Primary energy must exceed eCut.");

    particles_.clear();
    nextId_     = 0;
    stepCount_  = 0;
    primaryEnergy_ = energy;
    detector_.voxels.reset();  

    Particle primary;
    primary.id        = nextId();
    primary.parentId  = -1;         
    primary.type      = type;
    primary.energy    = energy;
    primary.position  = pos;
    primary.direction = dir.normalised();
    primary.status    = ParticleStatus::Active;


    primary.trackPoints.reserve(512);
    primary.trackPoints.push_back(pos);  

    particles_.push_back(std::move(primary));

    PSG_LOG_INFO("Primary seeded:", (int)type,
                 "| E0:", energy, "GeV",
                 "| t_max ~", math::showerMaximum(energy,
                               detector_.material.criticalEnergy),
                 "X0");
}


void ShowerSimulation::run()
{
    while (!isFinished() && stepCount_ < config_.maxSteps)
        step();

    if (stepCount_ >= config_.maxSteps)
        PSG_LOG_WARN("Hit maxSteps cap:", config_.maxSteps,
                     "— simulation may be incomplete.");

    PSG_LOG_INFO("Shower finished. Total particles:", totalCount(),
                 "| Steps:", stepCount_,
                 "| Energy deposited:",
                 detector_.totalDeposit(), "GeV");
}


void ShowerSimulation::step()
{
    const int n = static_cast<int>(particles_.size());
    for (int i = 0; i < n; ++i)
    {
        if (particles_[i].status == ParticleStatus::Active)
            stepParticle(particles_[i]);
    }
    ++stepCount_;
}

bool ShowerSimulation::isFinished() const noexcept
{
    return activeCount() == 0;
}

int ShowerSimulation::activeCount() const noexcept
{
    int count = 0;
    for (const auto& p : particles_)
        if (p.status == ParticleStatus::Active) ++count;
    return count;
}

void ShowerSimulation::stepParticle(Particle& p)
{
    const float ds = config_.stepSize;

    p.position = p.position + p.direction * ds;
    p.totalPathLength += ds;


    if (!detector_.contains(p.position.x, p.position.y, p.position.z))
    {
        p.status = ParticleStatus::Escaped;
        p.trackPoints.push_back(p.position);
        return;
    }


    if (p.totalPathLength > detector_.maxPathLength())
    {
        p.status = ParticleStatus::Escaped;
        return;
    }

    const float dE = computeDEdx(p) * ds;

    // Clamp: can't lose more energy than the particle has
    const float dE_clamped = std::min(dE, p.energy);

    p.energy              -= dE_clamped;
    p.totalEnergyDeposit  += dE_clamped;

    detector_.depositEnergy(p.position.x, p.position.y, p.position.z,
                             dE_clamped);

 
    if (p.energy < config_.eCut)
    {
        p.status = ParticleStatus::Stopped;
        p.trackPoints.push_back(p.position);
        return;
    }


    if (static_cast<int>(particles_.size()) < config_.maxParticles)
    {
        const bool isEM = (p.type == ParticleType::Electron ||
                           p.type == ParticleType::Positron);
        const bool isPhoton = (p.type == ParticleType::Photon);


        const float minBranchE = 2.0f * config_.eCut;

        if (isEM && p.energy > minBranchE && sampleBrem(p))
        {
       
            branchFrom(p, p.type, ParticleType::Photon);
        }
        else if (isPhoton && p.energy > 2.0f * math::Me && samplePair(p))
        {
         
            branchFrom(p, ParticleType::Electron, ParticleType::Positron);
        }
    }


    if (stepCount_ % 2 == 0)
        p.trackPoints.push_back(p.position);
}


 
float ShowerSimulation::computeDEdx(const Particle& p) const noexcept
{
    if (p.type == ParticleType::Photon)
        return 0.0f;

    const float X0 = detector_.material.radiationLength;
    const float Ec = detector_.material.criticalEnergy;

    return std::max(p.energy, Ec) / X0;
}


bool ShowerSimulation::sampleBrem(const Particle& p)
{
    const float ds = config_.stepSize;
    const float X0 = detector_.material.radiationLength;

    const float prob = 1.0f - std::exp(-ds / X0);
    return uniform01_(rng_) < prob;
}


bool ShowerSimulation::samplePair(const Particle& p)
{
    const float ds = config_.stepSize;
    const float X0 = detector_.material.radiationLength;

    const float prob = 1.0f - std::exp(-7.0f * ds / (9.0f * X0));
    return uniform01_(rng_) < prob;
}

void ShowerSimulation::branchFrom(Particle& parent,
                                  ParticleType d1Type,
                                  ParticleType d2Type)
{
    const bool isPairProd = (parent.type == ParticleType::Photon);
    const float parentE   = parent.energy;
    const float splitFrac = config_.splitFrac;

    const float E1 = parentE * splitFrac;
    const float E2 = parentE * (1.0f - splitFrac);

   
    if (E1 < config_.eCut || E2 < config_.eCut)
        return;


    const float theta = math::Me / std::max(parentE, math::Me);

    const float cosT = std::cos(theta);
    const float sinT = std::sin(theta);

    Vec3 dir1 = {
         parent.direction.x * cosT + parent.direction.z * sinT,
         parent.direction.y,
        -parent.direction.x * sinT + parent.direction.z * cosT
    };
    Vec3 dir2 = {
         parent.direction.x * cosT - parent.direction.z * sinT,
         parent.direction.y,
         parent.direction.x * sinT + parent.direction.z * cosT
    };

    Particle daughter1;
    daughter1.id       = nextId();
    daughter1.parentId = parent.id;
    daughter1.type     = d1Type;
    daughter1.energy   = E1;
    daughter1.position = parent.position;
    daughter1.direction = dir1.normalised();
    daughter1.status   = ParticleStatus::Active;
    daughter1.trackPoints.reserve(256);
    daughter1.trackPoints.push_back(parent.position);


    Particle daughter2;
    daughter2.id       = nextId();
    daughter2.parentId = parent.id;
    daughter2.type     = d2Type;
    daughter2.energy   = E2;
    daughter2.position = parent.position;
    daughter2.direction = dir2.normalised();
    daughter2.status   = ParticleStatus::Active;
    daughter2.trackPoints.reserve(256);
    daughter2.trackPoints.push_back(parent.position);


    if (isPairProd)
    {
        
        parent.energy = 0.0f;
        parent.status = ParticleStatus::Decayed;
    }
    else
    {
        parent.energy = E1;

      
        particles_.push_back(std::move(daughter2));
        return;
    }

   
    particles_.push_back(std::move(daughter1));
    particles_.push_back(std::move(daughter2));
}

} 
