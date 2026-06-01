#pragma once
#include <algorithm>
#include <vector>

namespace psg {

struct VoxelGrid
{
    int nx;
    int ny;
    int nz;
    std::vector<float> data;
    float totalEnergy = 0.0f;

    VoxelGrid(int x = 20, int y = 20, int z = 40)
        : nx(x), ny(y), nz(z), data(x * y * z, 0.0f)
    {}

    void reset() noexcept
    {
        std::fill(data.begin(), data.end(), 0.0f);
        totalEnergy = 0.0f;
    }

    void deposit(int ix, int iy, int iz, float dE) noexcept
    {
        data[static_cast<size_t>(ix * ny * nz + iy * nz + iz)] += dE;
        totalEnergy += dE;
    }

    [[nodiscard]] float total() const noexcept { return totalEnergy; }
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
        const int ix = std::clamp(static_cast<int>((x + halfX) / (2.0f * halfX) * voxels.nx), 0, voxels.nx - 1);
        const int iy = std::clamp(static_cast<int>((y + halfY) / (2.0f * halfY) * voxels.ny), 0, voxels.ny - 1);
        const int iz = std::clamp(static_cast<int>((z + halfZ) / (2.0f * halfZ) * voxels.nz), 0, voxels.nz - 1);
        voxels.deposit(ix, iy, iz, dE);
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