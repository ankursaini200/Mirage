find_package(GTest QUIET)

if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
    add_executable(mirage_tests
        test_version.cpp
        test_cpu_features.cpp
        test_image_loader.cpp
        test_color_converter.cpp
        test_tile_manager.cpp
    )

    target_link_libraries(mirage_tests
        PRIVATE
            mirage::mirage_core
            GTest::gtest
            GTest::gtest_main
    )

    target_compile_features(mirage_tests PRIVATE cxx_std_20)

    add_test(NAME mirage_unit_tests COMMAND mirage_tests)
else()
    message(WARNING "GoogleTest not found. Unit test binary 'mirage_tests' will not be built.")
endif()

# CLI Integration Tests executed via CTest
if(TARGET mirage)
    add_test(NAME cli_test_help COMMAND mirage --help)
    add_test(NAME cli_test_version COMMAND mirage --version)
    add_test(NAME cli_test_info COMMAND mirage --info)
    add_test(NAME cli_test_invalid_option COMMAND mirage --invalid-flag)
    set_tests_properties(cli_test_invalid_option PROPERTIES WILL_FAIL TRUE)

    add_test(NAME cli_test_image_info_missing COMMAND mirage --image-info)
    set_tests_properties(cli_test_image_info_missing PROPERTIES WILL_FAIL TRUE)

    add_test(NAME cli_test_tile_info_missing COMMAND mirage --tile-info dummy.png)
    set_tests_properties(cli_test_tile_info_missing PROPERTIES WILL_FAIL TRUE)
endif()
