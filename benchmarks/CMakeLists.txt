# Google Benchmark target (if installed on the host)
find_package(benchmark QUIET)

if(TARGET benchmark::benchmark)
    add_executable(mirage_google_benchmarks
        benchmark_cpu.cpp
    )

    target_link_libraries(mirage_google_benchmarks
        PRIVATE
            mirage::mirage_core
            benchmark::benchmark
            benchmark::benchmark_main
    )

    target_compile_features(mirage_google_benchmarks PRIVATE cxx_std_20)
endif()

# Standalone high-precision benchmark runner (requires only standard C++20 and mirage_core)
add_executable(mirage_benchmarks
    benchmark_runner.cpp
)

target_link_libraries(mirage_benchmarks
    PRIVATE
        mirage::mirage_core
)

target_compile_features(mirage_benchmarks PRIVATE cxx_std_20)
