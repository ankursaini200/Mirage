#include <benchmark/benchmark.h>

#include "mirage/cpu/cpu_features.hpp"

static void BM_CpuFeaturesDetection(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(mirage::cpu::CpuFeatures::detect());
    }
}
BENCHMARK(BM_CpuFeaturesDetection);

BENCHMARK_MAIN();

