## Description

Brief summary of the changes introduced in this pull request and architectural rationale.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing behavior to change)
- [ ] Architecture / Refactoring (no functional change)
- [ ] Documentation update
- [ ] CI / Build system update

## Testing Performed

- [ ] Configured and compiled with GCC in Debug and Release mode
- [ ] Configured and compiled with Clang in Debug and Release mode
- [ ] Unit tests pass via CTest (`ctest --test-dir build --output-on-failure`)
- [ ] Verified CLI flags (`--help`, `--version`, `--info`)
- [ ] Hardware counter / microbenchmark data attached (if claiming performance changes)

## Checklist

- [ ] My code adheres to the project's coding standards and C++20 conventions
- [ ] Public headers are placed in `include/mirage/` and internal implementations in `src/`
- [ ] Safe CPU feature detection is maintained without executing unsanctioned SIMD instructions
- [ ] Tests covering new functionality have been added
- [ ] Documentation has been updated accordingly
- [ ] No unexecuted tests or fabricated metrics are claimed

