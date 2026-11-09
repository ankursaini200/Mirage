#include <iostream>
#include <string_view>

#include "mirage/cpu/cpu_features.hpp"
#include "mirage/version.hpp"

namespace {

void print_help(std::string_view program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n"
              << "Mirage: " << mirage::PROJECT_DESCRIPTION << "\n\n"
              << "Available Options:\n"
              << "  -v, --version  Display Mirage version information\n"
              << "  -i, --info     Display project, compiler, and hardware environment details\n"
              << "  -h, --help     Display this help message\n";
}

void print_version() {
    std::cout << mirage::PROJECT_NAME << " version " << mirage::get_version_string() << "\n";
}

void print_info() {
    const auto& cpu = mirage::cpu::CpuFeatures::detect();

    std::cout << "Project name:              " << mirage::PROJECT_NAME << "\n"
              << "Project version:           " << mirage::get_version_string() << "\n"
              << "C++ standard:              " << mirage::get_cpp_standard_string() << "\n"
              << "Compiler information:      " << mirage::get_compiler_info_string() << "\n"
              << "Detected CPU architecture: " << cpu.architecture_name() << "\n"
              << "AVX2 availability:         " << cpu.avx2_status_string() << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }

    const std::string_view arg{argv[1]};

    if (arg == "--version" || arg == "-v") {
        print_version();
        return 0;
    }

    if (arg == "--info" || arg == "-i") {
        print_info();
        return 0;
    }

    if (arg == "--help" || arg == "-h") {
        print_help(argv[0]);
        return 0;
    }

    std::cerr << "Error: Unrecognized option '" << arg << "'\n\n";
    print_help(argv[0]);
    return 1;
}

