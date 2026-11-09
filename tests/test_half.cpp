#include <gtest/gtest.h>
#include <cmath>
#include <limits>

#include "mirage/core/half.hpp"

namespace mirage::tests {

using core::half;

TEST(HalfPrecisionTest, NormalValuesAndExactBitPatterns) {
    // Exact IEEE 754 binary16 bit patterns:
    // 0.0   = 0x0000
    // 1.0   = 0x3C00
    // -1.0  = 0xBC00
    // 2.0   = 0x4000
    // 0.5   = 0x3800
    // 65504 = 0x7BFF (max finite)
    // 2^-14 = 0x0400 (min positive normal = 6.103515625e-5)

    EXPECT_EQ(half(0.0f).storage, 0x0000u);
    EXPECT_EQ(half(1.0f).storage, 0x3C00u);
    EXPECT_EQ(half(-1.0f).storage, 0xBC00u);
    EXPECT_EQ(half(2.0f).storage, 0x4000u);
    EXPECT_EQ(half(0.5f).storage, 0x3800u);
    EXPECT_EQ(half(65504.0f).storage, 0x7BFFu);
    EXPECT_EQ(half(6.103515625e-5f).storage, 0x0400u);

    // Float roundtrip
    EXPECT_EQ(half(1.0f).to_float(), 1.0f);
    EXPECT_EQ(half(-1.0f).to_float(), -1.0f);
    EXPECT_EQ(half(0.5f).to_float(), 0.5f);
    EXPECT_EQ(half(65504.0f).to_float(), 65504.0f);
}

TEST(HalfPrecisionTest, SignedZeroTreatment) {
    const half pos_zero(0.0f);
    const half neg_zero(-0.0f);

    EXPECT_EQ(pos_zero.storage, 0x0000u);
    EXPECT_EQ(neg_zero.storage, 0x8000u);

    EXPECT_FALSE(pos_zero.signbit());
    EXPECT_TRUE(neg_zero.signbit());

    EXPECT_TRUE(pos_zero.is_zero());
    EXPECT_TRUE(neg_zero.is_zero());

    // In IEEE 754, +0.0 == -0.0 must evaluate to true
    EXPECT_TRUE(pos_zero == neg_zero);
    EXPECT_FALSE(pos_zero != neg_zero);

    // Float conversions preserve sign
    EXPECT_FALSE(std::signbit(pos_zero.to_float()));
    EXPECT_TRUE(std::signbit(neg_zero.to_float()));
}

TEST(HalfPrecisionTest, SubnormalValues) {
    // Minimum positive subnormal: 2^-24 = 5.9604644775390625e-8 -> 0x0001
    const half min_subnormal(5.9604644775390625e-8f);
    EXPECT_EQ(min_subnormal.storage, 0x0001u);
    EXPECT_TRUE(min_subnormal.is_subnormal());
    EXPECT_FALSE(min_subnormal.is_zero());
    EXPECT_NEAR(min_subnormal.to_float(), 5.9604644775390625e-8f, 1e-15f);

    // Mid-range subnormal: 2^-15 = 3.0517578125e-5 -> 0x0200
    const half mid_subnormal(3.0517578125e-5f);
    EXPECT_EQ(mid_subnormal.storage, 0x0200u);
    EXPECT_TRUE(mid_subnormal.is_subnormal());
    EXPECT_NEAR(mid_subnormal.to_float(), 3.0517578125e-5f, 1e-10f);
}

TEST(HalfPrecisionTest, RoundToNearestTiesToEven) {
    // In binary16, 1.0 is 0x3C00 (mantissa 0, LSB = 0, even).
    // The next representable binary16 value is 1.0 + 2^-10 (0x3C01, LSB = 1, odd).
    // The next after that is 1.0 + 2^-9 (0x3C02, LSB = 0, even).
    //
    // Case 1: Exact tie between 0x3C00 (even) and 0x3C01 (odd).
    // Midpoint = 1.0 + 2^-11.
    // Under ties-to-even, the midpoint must round down to 0x3C00 (even):
    const float tie_to_even = 1.0f + std::ldexp(1.0f, -11);
    EXPECT_EQ(half(tie_to_even).storage, 0x3C00u);

    // Case 2: Exact tie between 0x3C01 (odd) and 0x3C02 (even).
    // Midpoint = 1.0 + 2^-10 + 2^-11.
    // Under ties-to-even, this midpoint must round up to 0x3C02 (even):
    const float tie_to_odd_rounds_to_even = 1.0f + std::ldexp(1.0f, -10) + std::ldexp(1.0f, -11);
    EXPECT_EQ(half(tie_to_odd_rounds_to_even).storage, 0x3C02u);
}

TEST(HalfPrecisionTest, OverflowAndUnderflow) {
    // Overflow beyond 65504.0 rounds to infinity
    const half overflow_pos(70000.0f);
    EXPECT_TRUE(overflow_pos.is_inf());
    EXPECT_EQ(overflow_pos.storage, 0x7C00u);

    const half overflow_neg(-70000.0f);
    EXPECT_TRUE(overflow_neg.is_inf());
    EXPECT_EQ(overflow_neg.storage, 0xFC00u);

    // Underflow below 2^-25 rounds to signed zero
    const half underflow_pos(1.0e-9f);
    EXPECT_TRUE(underflow_pos.is_zero());
    EXPECT_EQ(underflow_pos.storage, 0x0000u);

    const half underflow_neg(-1.0e-9f);
    EXPECT_TRUE(underflow_neg.is_zero());
    EXPECT_EQ(underflow_neg.storage, 0x8000u);
}

TEST(HalfPrecisionTest, InfinityAndNaN) {
    const half pos_inf(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(pos_inf.is_inf());
    EXPECT_FALSE(pos_inf.is_nan());
    EXPECT_EQ(pos_inf.storage, 0x7C00u);

    const half neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_TRUE(neg_inf.is_inf());
    EXPECT_EQ(neg_inf.storage, 0xFC00u);

    const half nan_val(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(nan_val.is_nan());
    EXPECT_FALSE(nan_val.is_inf());
    EXPECT_FALSE(nan_val == nan_val); // NaN != NaN
}

TEST(HalfPrecisionTest, EmulatedArithmeticOperators) {
    const half a(1.5f);
    const half b(2.5f);

    EXPECT_NEAR((a + b).to_float(), 4.0f, 1e-4f);
    EXPECT_NEAR((b - a).to_float(), 1.0f, 1e-4f);
    EXPECT_NEAR((a * b).to_float(), 3.75f, 1e-4f);
    EXPECT_NEAR((b / a).to_float(), 1.66667f, 1e-3f);

    half c(2.0f);
    c += half(3.0f);
    EXPECT_NEAR(c.to_float(), 5.0f, 1e-4f);
}

}  // namespace mirage::tests
