#pragma once

#include <cmath>
#include <cstdint>
#include <string>

namespace mirage::core {

/**
 * @brief IEEE 754 binary16 (half-precision) floating-point representation.
 *
 * Implements:
 * - 16-bit storage (1 sign, 5 exponent with bias 15, 10 fraction).
 * - Exact round-to-nearest, ties-to-even (RNTE) conversion.
 * - Full support for normal numbers, subnormals, signed zeros (+0/-0), infinities, and NaNs.
 * - Explicit binary16 emulation: arithmetic operations promote to float32 and round
 *   the intermediate results back to binary16.
 */
class half {
public:
    uint16_t storage{0};

    constexpr half() noexcept = default;
    constexpr explicit half(uint16_t raw_bits) noexcept : storage(raw_bits) {}
    explicit half(float val) noexcept;

    [[nodiscard]] float to_float() const noexcept;
    explicit operator float() const noexcept { return to_float(); }

    // Arithmetic operators with explicit binary16 rounding
    [[nodiscard]] half operator+(half rhs) const noexcept { return half(to_float() + rhs.to_float()); }
    [[nodiscard]] half operator-(half rhs) const noexcept { return half(to_float() - rhs.to_float()); }
    [[nodiscard]] half operator*(half rhs) const noexcept { return half(to_float() * rhs.to_float()); }
    [[nodiscard]] half operator/(half rhs) const noexcept { return half(to_float() / rhs.to_float()); }

    half& operator+=(half rhs) noexcept { *this = *this + rhs; return *this; }
    half& operator-=(half rhs) noexcept { *this = *this - rhs; return *this; }
    half& operator*=(half rhs) noexcept { *this = *this * rhs; return *this; }
    half& operator/=(half rhs) noexcept { *this = *this / rhs; return *this; }

    [[nodiscard]] half operator-() const noexcept {
        return half(static_cast<uint16_t>(storage ^ 0x8000u));
    }

    [[nodiscard]] bool is_nan() const noexcept {
        return (storage & 0x7C00u) == 0x7C00u && (storage & 0x03FFu) != 0;
    }

    [[nodiscard]] bool is_inf() const noexcept {
        return (storage & 0x7FFFu) == 0x7C00u;
    }

    [[nodiscard]] bool is_zero() const noexcept {
        return (storage & 0x7FFFu) == 0;
    }

    [[nodiscard]] bool is_subnormal() const noexcept {
        return (storage & 0x7C00u) == 0 && (storage & 0x03FFu) != 0;
    }

    [[nodiscard]] bool signbit() const noexcept {
        return (storage & 0x8000u) != 0;
    }

    [[nodiscard]] bool operator==(half rhs) const noexcept;
    [[nodiscard]] bool operator!=(half rhs) const noexcept { return !(*this == rhs); }
    [[nodiscard]] bool operator<(half rhs) const noexcept;
    [[nodiscard]] bool operator<=(half rhs) const noexcept { return (*this < rhs) || (*this == rhs); }
    [[nodiscard]] bool operator>(half rhs) const noexcept { return rhs < *this; }
    [[nodiscard]] bool operator>=(half rhs) const noexcept { return !(*this < rhs); }

    [[nodiscard]] static constexpr half zero(bool negative = false) noexcept {
        return half(static_cast<uint16_t>(negative ? 0x8000u : 0x0000u));
    }
    [[nodiscard]] static constexpr half one() noexcept { return half(static_cast<uint16_t>(0x3C00u)); }
    [[nodiscard]] static constexpr half infinity(bool negative = false) noexcept {
        return half(static_cast<uint16_t>(negative ? 0xFC00u : 0x7C00u));
    }
    [[nodiscard]] static constexpr half quiet_nan() noexcept { return half(static_cast<uint16_t>(0x7E00u)); }
    [[nodiscard]] static constexpr half min_positive_normal() noexcept { return half(static_cast<uint16_t>(0x0400u)); } // 2^-14 ~= 6.10e-5
    [[nodiscard]] static constexpr half min_positive_subnormal() noexcept { return half(static_cast<uint16_t>(0x0001u)); } // 2^-24 ~= 5.96e-8
    [[nodiscard]] static constexpr half max_finite() noexcept { return half(static_cast<uint16_t>(0x7BFFu)); } // 65504.0
};

/**
 * @brief Portable helper function: quantizes float32 value to IEEE 754 binary16 precision
 * and returns it as float32.
 */
[[nodiscard]] inline float round_to_fp16(float f) noexcept {
    return half(f).to_float();
}

}  // namespace mirage::core
