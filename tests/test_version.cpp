#include <gtest/gtest.h>

#include "mirage/version.hpp"

namespace mirage::tests {

TEST(VersionTest, SemanticVersionComponents) {
    EXPECT_GE(VERSION_MAJOR, 0);
    EXPECT_GE(VERSION_MINOR, 0);
    EXPECT_GE(VERSION_PATCH, 0);
    EXPECT_FALSE(VERSION_PRERELEASE.empty());
}

TEST(VersionTest, VersionStringFormatting) {
    const std::string version = get_version_string();
    EXPECT_FALSE(version.empty());

    // Should contain major version digit
    EXPECT_NE(version.find(std::to_string(VERSION_MAJOR)), std::string::npos);

    // Should contain prerelease identifier if set
    if (!VERSION_PRERELEASE.empty()) {
        EXPECT_NE(version.find(VERSION_PRERELEASE), std::string::npos);
    }
}

TEST(VersionTest, CppStandardIdentification) {
    const std::string standard = get_cpp_standard_string();
    EXPECT_FALSE(standard.empty());
    EXPECT_NE(standard.find("C++"), std::string::npos);
}

TEST(VersionTest, CompilerIdentification) {
    const std::string compiler = get_compiler_info_string();
    EXPECT_FALSE(compiler.empty());
}

TEST(VersionTest, ProjectMetadata) {
    EXPECT_EQ(PROJECT_NAME, "Mirage");
    EXPECT_FALSE(PROJECT_DESCRIPTION.empty());
}

}  // namespace mirage::tests

