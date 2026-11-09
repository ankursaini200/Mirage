#pragma once

#include <stdexcept>
#include <string>

namespace mirage::core {

/**
 * @brief Base exception class for all Mirage runtime errors.
 */
class MirageException : public std::runtime_error {
public:
    explicit MirageException(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @brief Exception thrown when image or tile dimensions are invalid (e.g. non-positive or exceeding limits).
 */
class InvalidDimensionException : public MirageException {
public:
    explicit InvalidDimensionException(const std::string& message)
        : MirageException("InvalidDimensionException: " + message) {}
};

/**
 * @brief Exception thrown when accessing coordinates or pixels outside valid boundaries.
 */
class OutOfBoundsException : public MirageException {
public:
    explicit OutOfBoundsException(const std::string& message)
        : MirageException("OutOfBoundsException: " + message) {}
};

/**
 * @brief Exception thrown when an unsupported or mismatched color format is encountered.
 */
class UnsupportedColorFormatException : public MirageException {
public:
    explicit UnsupportedColorFormatException(const std::string& message)
        : MirageException("UnsupportedColorFormatException: " + message) {}
};

/**
 * @brief Exception thrown when memory allocation fails or alignment requirements cannot be met.
 */
class AllocationException : public MirageException {
public:
    explicit AllocationException(const std::string& message)
        : MirageException("AllocationException: " + message) {}
};

}  // namespace mirage::core
