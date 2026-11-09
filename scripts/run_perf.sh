#!/usr/bin/env bash
# Mirage Phase 8: External Linux perf stat wrapper script
# Collects hardware performance counters via the system 'perf' CLI tool.

set -euo pipefail

BENCH_BIN="build/release/benchmarks/mirage_benchmarks"

if [[ ! -x "${BENCH_BIN}" ]]; then
    echo "Error: Benchmark binary '${BENCH_BIN}' not found."
    echo "Please build with: cmake --build build/release --target mirage_benchmarks"
    exit 1
fi

echo "================================================================================"
echo "                   MIRAGE HARDWARE PROFILER (LINUX PERF)"
echo "================================================================================"

# Check if 'perf' command-line binary is installed
if ! command -v perf &> /dev/null; then
    echo "Notice: System CLI 'perf' utility is not installed in PATH."
    echo ""
    echo "To install 'perf' on Arch Linux / Manjaro:"
    echo "  sudo pacman -S perf"
    echo "To install on Ubuntu / Debian:"
    echo "  sudo apt-get install linux-tools-generic linux-tools-\$(uname -r)"
    echo "To install on Fedora / RHEL:"
    echo "  sudo dnf install perf"
    echo ""
    echo "Security & Permission Settings:"
    echo "Current /proc/sys/kernel/perf_event_paranoid level:"
    if [[ -f /proc/sys/kernel/perf_event_paranoid ]]; then
        cat /proc/sys/kernel/perf_event_paranoid
    else
        echo "unavailable"
    fi
    echo ""
    echo "To allow unprivileged user-space hardware counter sampling across all processes:"
    echo "  sudo sysctl -w kernel.perf_event_paranoid=1"
    echo "To allow full kernel + user profiling:"
    echo "  sudo sysctl -w kernel.perf_event_paranoid=-1"
    echo ""
    echo "Note: Mirage already includes native, in-process hardware counter sampling via the"
    echo "Linux perf_event_open() syscall, which runs automatically without requiring 'perf' CLI!"
    echo "Running native benchmark directly..."
    echo "================================================================================"
    "${BENCH_BIN}" --quick
    exit 0
fi

# If perf CLI is installed, check paranoid level
PARANOID=2
if [[ -f /proc/sys/kernel/perf_event_paranoid ]]; then
    PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid)
fi

echo "Host CPU: $(lscpu | grep 'Model name' | sed 's/Model name:[ \t]*//')"
echo "Kernel:   $(uname -r)"
echo "perf_event_paranoid: ${PARANOID}"
echo ""

# Hardware events to profile
EVENTS="cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock"

echo "Executing: perf stat -r 5 -e ${EVENTS} ${BENCH_BIN} --quick"
echo "================================================================================"
perf stat -r 5 -e "${EVENTS}" "${BENCH_BIN}" --quick

