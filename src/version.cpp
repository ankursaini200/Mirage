#include "mirage/version.hpp"

#include <sstream>

namespace mirage {

std::string get_version_string() {
    std::ostringstream oss;
    oss << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH;
    if (!VERSION_PRERELEASE.empty()) {
        oss << "-" << VERSION_PRERELEASE;
    }
    return oss.str();
}

std::string get_cpp_standard_string() {
#if defined(_MSVC_LANG)
    const long std_ver = _MSVC_LANG;
#elif defined(__cplusplus)
    const long std_ver = __cplusplus;
#else
    const long std_ver = 0;
#endif

    if (std_ver == 202002L) {
        return "C++20 (202002L)";
    } else if (std_ver > 202002L) {
        return "C++23 or later (" + std::to_string(std_ver) + ")";
    } else if (std_ver == 201703L) {
        return "C++17 (201703L)";
    } else {
        return "Standard code " + std::to_string(std_ver);
    }
}

std::string get_compiler_info_string() {
    std::ostringstream oss;
#if defined(__clang__)
    oss << "Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(__GNUC__) || defined(__GNUG__)
    oss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#elif defined(_MSC_VER)
    oss << "MSVC " << _MSC_VER;
#else
    oss << "Unknown Compiler";
#endif
    return oss.str();
}

}  // namespace mirage

