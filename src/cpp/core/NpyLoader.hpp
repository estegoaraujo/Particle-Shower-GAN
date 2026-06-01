#pragma once

#include "DetectorVolume.hpp"
#include "Particle.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>

namespace psg {

class NpyLoader
{
public:
    static VoxelGrid load(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("NpyLoader: cannot open: " + path);

        uint8_t magic[6];
        file.read(reinterpret_cast<char*>(magic), 6);
        if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
            magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y')
            throw std::runtime_error("NpyLoader: not a .npy file: " + path);

        uint8_t major, minor;
        file.read(reinterpret_cast<char*>(&major), 1);
        file.read(reinterpret_cast<char*>(&minor), 1);

        uint16_t headerLen;
        file.read(reinterpret_cast<char*>(&headerLen), 2);

        std::string header(headerLen, ' ');
        file.read(header.data(), headerLen);

        VoxelGrid grid(20, 20, 40);
        file.read(reinterpret_cast<char*>(grid.data.data()),
                  static_cast<std::streamsize>(
                      grid.data.size() * sizeof(float)));

        if (!file.good())
            throw std::runtime_error("NpyLoader: read error: " + path);

        return grid;
    }

    static std::vector<Particle> voxelsToParticles(
        const VoxelGrid&      grid,
        const DetectorVolume& detector,
        float                 threshold   = 0.001f,
        int                   maxParticles = 2000,
        int                   idOffset    = 0)
    {
        std::vector<Particle> particles;

        const float dx = (2.0f * detector.halfX) / static_cast<float>(grid.nx);
        const float dy = (2.0f * detector.halfY) / static_cast<float>(grid.ny);
        const float dz = (2.0f * detector.halfZ) / static_cast<float>(grid.nz);

        float maxE = 0.0f;
        for (float v : grid.data)
            maxE = std::max(maxE, v);

        if (maxE < 1e-8f) return particles;

        int count = 0;
        for (int iz = 0; iz < grid.nz && count < maxParticles; ++iz)
        {
            for (int ix = 0; ix < grid.nx && count < maxParticles; ++ix)
            {
                for (int iy = 0; iy < grid.ny && count < maxParticles; ++iy)
                {
                    const float e = grid.data[static_cast<size_t>(
                        iz * grid.nx * grid.ny + ix * grid.ny + iy)] / maxE;

                    if (e < threshold) continue;

                    const float wx = -detector.halfX + (ix + 0.5f) * dx;
                    const float wy = -detector.halfY + (iy + 0.5f) * dy;
                    const float wz = -detector.halfZ + (iz + 0.5f) * dz;

                    Particle p;
                    p.id     = idOffset + count++;
                    p.type   = ParticleType::Electron;
                    p.status = ParticleStatus::Stopped;
                    p.energy = e;
                    p.position  = {wx, wy, wz};
                    p.direction = {0.0f, 0.0f, 1.0f};

                    p.trackPoints.push_back({wx, wy, wz - dz * 2.0f});
                    p.trackPoints.push_back({wx, wy, wz});
                    p.trackPoints.push_back({wx, wy, wz + dz * 2.0f});

                    particles.push_back(std::move(p));
                }
            }
        }

        return particles;
    }

    static std::vector<Particle> loadDirectory(
        const std::string&    dirPath,
        const DetectorVolume& detector,
        float                 threshold    = 0.001f,
        int                   maxPerSample = 200)
    {
        std::vector<Particle> all;
        int idOffset = 0;

        std::vector<std::filesystem::path> files;
        for (const auto& entry :
             std::filesystem::directory_iterator(dirPath))
        {
            if (entry.path().extension() == ".npy")
                files.push_back(entry.path());
        }

        std::sort(files.begin(), files.end());

        for (const auto& f : files)
        {
            try
            {
                VoxelGrid grid = load(f.string());
                auto particles = voxelsToParticles(
                    grid, detector, threshold, maxPerSample, idOffset);
                idOffset += static_cast<int>(particles.size());
                all.insert(all.end(),
                            std::make_move_iterator(particles.begin()),
                            std::make_move_iterator(particles.end()));
            }
            catch (const std::exception& e)
            {
                continue;
            }
        }

        return all;
    }
};

}
