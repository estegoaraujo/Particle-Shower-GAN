#pragma once

#include "DetectorVolume.hpp"
#include "ShowerSimulation.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace psg {

class Exporter
{
public:
    static void saveVoxelGrid(const VoxelGrid&   grid,
                               const std::string& filepath)
    {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Exporter: cannot open: " + filepath);

        const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
        file.write(reinterpret_cast<const char*>(magic), sizeof(magic));

        std::ostringstream headerStream;
        headerStream << "{'descr': '<f4', 'fortran_order': False, "
                     << "'shape': ("
                     << grid.nx << ", "
                     << grid.ny << ", "
                     << grid.nz << "), }";

        std::string header = headerStream.str();

        const size_t prefixBase = 10;
        size_t headerLen = header.size() + 1;
        size_t total     = prefixBase + headerLen;
        size_t remainder = total % 64;
        if (remainder != 0)
        {
            size_t padding = 64 - remainder;
            header.append(padding, ' ');
            headerLen += padding;
        }
        header += '\n';
        headerLen = header.size();

        const uint16_t hlen16 = static_cast<uint16_t>(headerLen);
        file.write(reinterpret_cast<const char*>(&hlen16), sizeof(hlen16));
        file.write(header.c_str(), static_cast<std::streamsize>(headerLen));

        file.write(
            reinterpret_cast<const char*>(grid.data.data()),
            static_cast<std::streamsize>(grid.data.size() * sizeof(float))
        );

        if (!file.good())
            throw std::runtime_error("Exporter: write error: " + filepath);
    }

    static void generateDataset(const std::string& outputDir,
                                  int                n,
                                  float              energy,
                                  ParticleType       type,
                                  uint32_t           baseSeed = 1)
    {
        const std::string csvPath = outputDir + "/metadata.csv";
        std::ofstream csv(csvPath);
        if (!csv.is_open())
            throw std::runtime_error("Exporter: cannot open: " + csvPath);

        csv << "shower_id,seed,primary_type,primary_energy_GeV,"
               "total_particles,total_deposit_GeV\n";

        SimConfig config;
        config.stepSize     = 0.5f;
        config.eCut         = 0.001f;
        config.maxParticles = 20000;

        DetectorVolume detector;

        for (int i = 0; i < n; ++i)
        {
            config.seed = baseSeed + static_cast<uint32_t>(i);

            detector.voxels.reset();

            ShowerSimulation sim(config, detector);
            sim.seedPrimary(type, energy,
                            {0.0f, 0.0f, -170.0f},
                            {0.0f, 0.0f,  1.0f});
            sim.run();

            char filename[64];
            std::snprintf(filename, sizeof(filename), "/shower_%04d.npy", i);
            const std::string path = outputDir + filename;

            saveVoxelGrid(detector.voxels, path);

            csv << i << ","
                << config.seed << ","
                << static_cast<int>(type) << ","
                << energy << ","
                << sim.totalCount() << ","
                << detector.totalDeposit() << "\n";

            if ((i + 1) % 100 == 0 || i == n - 1)
                std::cout << "[INFO]  Generated " << (i+1) << "/" << n << " showers\n";
        }

        std::cout << "[INFO]  Dataset saved to " << outputDir
                  << "  (" << n << " showers)\n";
    }
};

}
