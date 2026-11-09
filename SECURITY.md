# Security Policy

## Supported Versions

Mirage is currently in early research and active development (0.1.0-alpha). Security updates are applied to the latest development branch (`main`).

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Security Considerations for Research Software

Mirage processes image data and is intended for research in software-based image enhancement and computer architecture. In image processing software, untrusted image files can potentially carry malicious payloads or trigger vulnerabilities (e.g., buffer overflows, integer overflows in dimensions, or out-of-bounds reads/writes).

Our engineering principles include:
1. **Input Validation**: Strict validation of image dimensions, channel counts, and buffer boundaries before any memory allocation or SIMD processing.
2. **Bounds Checking**: Memory operations use validated strides and offsets.
3. **Safe CPU Dispatch**: No SIMD instructions (e.g., AVX2) are executed without verifying both hardware capability and operating system support (OSXSAVE / XCR0).
4. **No Arbitrary Memory Discards**: Unsupported pixel formats or corrupted headers fail fast and explicitly rather than silently processing undefined data.

## Reporting a Vulnerability

If you discover a security vulnerability in Mirage:

1. **Do not open a public GitHub issue.**
2. Send an email to the project maintainers detailing the vulnerability, including:
   - Description of the issue.
   - Steps to reproduce or a proof of concept (e.g., a crafted test file).
   - Affected system architecture, compiler, and OS.
   - Any suggested remediations or patches.
3. You will receive an acknowledgment within 48 hours.
4. Maintainers will coordinate a fix, verify it against test suites, and publish an advisory alongside a patched release.

