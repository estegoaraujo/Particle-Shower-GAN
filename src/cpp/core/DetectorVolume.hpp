#pragma once
#include <algorithm>

namespace psg {

struct VoxelGrid
{
    float totalEnergy = 0.0f;

    void  reset()                                                      noexcept { totalEnergy = 0.0f; }
    void  deposit(float /*x*/, float /*y*/, float /*z*/, float dE)    noexcept { totalEnergy += dE; }
    [[nodiscard]] float total()                                  const noexcept { return totalEnergy; }
};

struct Material
{
    float radiationLength    = 14.0f;   
    float interactionLength  = 83.7f;   
    float density            =  1.40f;  
    float criticalEnergy     =  0.032f; 
                                        
    const char* name = "Liquid Argon";
};


class DetectorVolume
{
public:
    float halfX = 100.0f;   
    float halfY = 100.0f;  
    float halfZ = 175.0f;  

    Material  material;
    VoxelGrid voxels;

    void depositEnergy(float x, float y, float z, float dE) noexcept
    {
        voxels.deposit(x, y, z, dE);
    }

    [[nodiscard]] float totalDeposit() const noexcept { return voxels.total(); }

    [[nodiscard]] bool contains(float x, float y, float z) const noexcept
    {
        return (x > -halfX && x < halfX)
            && (y > -halfY && y < halfY)
            && (z > -halfZ && z < halfZ);
    }

    [[nodiscard]] float maxPathLength() const noexcept
    {
        return 2.0f * std::max({halfX, halfY, halfZ}) * 1.733f; 
    }
};

}