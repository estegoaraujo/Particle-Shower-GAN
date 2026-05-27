#pragma once
/**
 * @file utils/Logger.hpp
 * @brief Minimal compile-time-tagged console logger.
 *
 * Usage:
 *   PSG_LOG_INFO("Simulation step", step, "active particles:", n);
 *   PSG_LOG_WARN("Energy below threshold — stopping particle", id);
 *   PSG_LOG_ERROR("GLAD failed to load OpenGL");
 */

#include <iostream>
#include <string_view>

namespace psg::log {

enum class Level { INFO, WARN, ERR };

inline constexpr std::string_view levelTag(Level l) {
    switch (l) {
        case Level::INFO: return "[INFO] ";
        case Level::WARN: return "[WARN] ";
        case Level::ERR:  return "[ERR]  ";
    }
    return "[???]  ";
}

template<typename... Args>
void print(Level l, Args&&... args) {
    std::cout << levelTag(l);
    ((std::cout << args << ' '), ...);
    std::cout << '\n';
}

} // namespace psg::log

#define PSG_LOG_INFO(...)  psg::log::print(psg::log::Level::INFO,  __VA_ARGS__)
#define PSG_LOG_WARN(...)  psg::log::print(psg::log::Level::WARN,  __VA_ARGS__)
#define PSG_LOG_ERROR(...) psg::log::print(psg::log::Level::ERR,   __VA_ARGS__)
