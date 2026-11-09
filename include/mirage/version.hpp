#pragma once

#include <string>
#include <string_view>

namespace mirage {

inline constexpr int VERSION_MAJOR = 0;
inline constexpr int VERSION_MINOR = 1;
inline constexpr int VERSION_PATCH = 0;
inline constexpr std::string_view VERSION_PRERELEASE = "alpha";

inline constexpr std::string_view PROJECT_NAME = "Mirage";
inline constexpr std::string_view PROJECT_DESCRIPTION =
    "Adaptive Precision for Software-Based Color Image Enhancement";

/**
 * @brief Returns the complete semantic version string (e.g., "0.1.0-alpha").
 */
[[nodiscard]] std::string get_version_string();

/**
 * @brief Returns the C++ standard version string used during compilation.
 */
[[nodiscard]] std::string get_cpp_standard_string();

/**
 * @brief Returns the compiler identification and version string.
 */
[[nodiscard]] std::string get_compiler_info_string();

}  // namespace mirage

