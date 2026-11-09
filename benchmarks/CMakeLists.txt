find_package(benchmark QUIET)

if(TARGET benchmark::benchmark)
    add_executable(mirage_benchmarks
        benchmark_cpu.cpp
    )

    target_link_libraries(mirage_benchmarks
        PRIVATE
            mirage::mirage_core
            benchmark::benchmark
            benchmark::benchmark_main
    )

    target_compile_features(mirage_benchmarks PRIVATE cxx_std_20)
else()
    message(STATUS "Google Benchmark not found. Benchmarks will not be built.")
endif()

