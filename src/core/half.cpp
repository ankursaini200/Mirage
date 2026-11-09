#include "mirage/core/half.hpp"

#include <bit>
#include <cmath>

namespace mirage::core {

half::half(float val) noexcept {
    const uint32_t f_bits = std::bit_cast<uint32_t>(val);
    const uint32_t sign = (f_bits >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((f_bits >> 23) & 0xFFu);
    uint32_t mant = f_bits & 0x7FFFFFu;

    // Special values: Infinity or NaN (exp == 255)
    if (exp == 255) {
        if (mant == 0) {
            storage = static_cast<uint16_t>(sign | 0x7C00u); // Infinity
        } else {
            // Quiet NaN with non-zero payload
            const uint16_t nan_payload = static_cast<uint16_t>(mant >> 13);
            storage = static_cast<uint16_t>(sign | 0x7C00u | (nan_payload ? nan_payload : 0x0200u));
        }
        return;
    }

    // Zero or Underflow (exp < 113)
    if (exp < 113) {
        if (exp < 103) {
            // Complete underflow to signed zero
            storage = static_cast<uint16_t>(sign);
            return;
        }

        // Subnormal FP16: Add implicit leading 1 bit from normalized float
        mant |= 0x00800000u;
        const int32_t shift = 113 - exp;
        const int32_t total_shift = 13 + shift;

        // Round-to-nearest, ties-to-even
        const uint32_t round_bit = (mant >> (total_shift - 1)) & 1u;
        const uint32_t sticky_bits = mant & ((1u << (total_shift - 1)) - 1u);
        const uint32_t lsb = (mant >> total_shift) & 1u;

        uint32_t result_mant = mant >> total_shift;
        if (round_bit && (sticky_bits != 0 || lsb)) {
            result_mant++;
        }

        storage = static_cast<uint16_t>(sign | result_mant);
        return;
    }

    // Overflow to Infinity (exp > 142)
    if (exp > 142) {
        storage = static_cast<uint16_t>(sign | 0x7C00u);
        return;
    }

    // Normal FP16 (113 <= exp <= 142)
    int32_t exp16 = exp - 112;

    // Round-to-nearest, ties-to-even from 23 mantissa bits to 10
    const uint32_t round_bit = (mant >> 12) & 1u;
    const uint32_t sticky_bits = mant & 0x0FFFu;
    const uint32_t lsb = (mant >> 13) & 1u;

    uint32_t result_mant = mant >> 13;
    if (round_bit && (sticky_bits != 0 || lsb)) {
        result_mant++;
        if (result_mant >= 0x0400u) {
            result_mant = 0;
            exp16++;
            if (exp16 >= 31) {
                storage = static_cast<uint16_t>(sign | 0x7C00u); // Overflowed on rounding
                return;
            }
        }
    }

    storage = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp16) << 10) | result_mant);
}

float half::to_float() const noexcept {
    const uint32_t sign = (static_cast<uint32_t>(storage) & 0x8000u) << 16;
    const uint32_t exp16 = (static_cast<uint32_t>(storage) >> 10) & 0x1Fu;
    uint32_t mant16 = static_cast<uint32_t>(storage) & 0x03FFu;

    // Special values: Infinity or NaN
    if (exp16 == 31) {
        if (mant16 == 0) {
            return std::bit_cast<float>(sign | 0x7F800000u); // Infinity
        } else {
            return std::bit_cast<float>(sign | 0x7F800000u | (mant16 << 13)); // NaN
        }
    }

    // Zero or Subnormal
    if (exp16 == 0) {
        if (mant16 == 0) {
            return std::bit_cast<float>(sign); // Signed zero (+0.0f or -0.0f)
        }

        // Subnormal: Normalize mantissa
        int32_t shift = 0;
        while ((mant16 & 0x0400u) == 0) {
            mant16 <<= 1;
            shift++;
        }
        const uint32_t exp32 = static_cast<uint32_t>(113 - shift) << 23;
        const uint32_t mant32 = (mant16 & 0x03FFu) << 13;
        return std::bit_cast<float>(sign | exp32 | mant32);
    }

    // Normal FP16
    const uint32_t exp32 = (exp16 + 112u) << 23;
    const uint32_t mant32 = mant16 << 13;
    return std::bit_cast<float>(sign | exp32 | mant32);
}

bool half::operator==(half rhs) const noexcept {
    if (is_nan() || rhs.is_nan()) {
        return false;
    }
    if (is_zero() && rhs.is_zero()) {
        return true; // +0.0 == -0.0
    }
    return storage == rhs.storage;
}

bool half::operator<(half rhs) const noexcept {
    if (is_nan() || rhs.is_nan()) {
        return false;
    }
    if (is_zero() && rhs.is_zero()) {
        return false;
    }

    const bool s1 = signbit();
    const bool s2 = rhs.signbit();

    if (s1 != s2) {
        return s1; // Negative is smaller than positive
    }

    if (s1) {
        // Both negative: larger magnitude means smaller value
        return (storage & 0x7FFFu) > (rhs.storage & 0x7FFFu);
    } else {
        // Both positive
        return (storage & 0x7FFFu) < (rhs.storage & 0x7FFFu);
    }
}

}  // namespace mirage::core

