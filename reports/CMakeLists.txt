cmake_minimum_required(VERSION 3.20)

option(FROM_NATIVE_ARCH "Enable native CPU-specific optimizations such as -march=native" OFF)
option(FROM_CUDA "Enable CUDA backend when the CUDA Toolkit is installed" OFF)

project(from LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE FROM_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cli/*.cpp"
)

if(FROM_CUDA)
    file(GLOB FROM_WINDOWS_CUDA_ROOTS
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin"
    )
    find_program(FROM_NVCC nvcc
        PATHS ${FROM_WINDOWS_CUDA_ROOTS}
        NO_DEFAULT_PATH
    )
    if(NOT FROM_NVCC)
        find_program(FROM_NVCC nvcc)
    endif()
    if(FROM_NVCC)
        set(CMAKE_CUDA_COMPILER "${FROM_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
        enable_language(CUDA)
        get_filename_component(FROM_CUDA_BIN_DIR "${FROM_NVCC}" DIRECTORY)
        get_filename_component(FROM_CUDA_ROOT "${FROM_CUDA_BIN_DIR}" DIRECTORY)
        set(CUDAToolkit_ROOT "${FROM_CUDA_ROOT}" CACHE PATH "CUDA Toolkit root" FORCE)
        find_package(CUDAToolkit REQUIRED)
        file(GLOB_RECURSE FROM_CUDA_SOURCES CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu"
        )
        list(APPEND FROM_SOURCES ${FROM_CUDA_SOURCES})
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    else()
        message(WARNING "FROM_CUDA=ON requested, but nvcc/CUDA Toolkit was not found. Building CPU-only.")
        set(FROM_CUDA OFF)
    endif()
endif()

add_executable(from ${FROM_SOURCES})
target_include_directories(from PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(FROM_CUDA)
    target_compile_definitions(from PRIVATE FROM_CUDA=1)
    set_source_files_properties(${FROM_CUDA_SOURCES} PROPERTIES COMPILE_OPTIONS "")
    target_compile_options(from PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3;--use_fast_math;--generate-line-info;
            -Xcompiler=/O2;-Xcompiler=/EHsc
        >
    )
    target_link_libraries(from PRIVATE CUDA::cudart CUDA::cublas)
else()
    target_compile_definitions(from PRIVATE FROM_CPU_ONLY=1)
endif()

if(MSVC)
    target_compile_options(from PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/EHsc>
        $<$<COMPILE_LANGUAGE:C>:/W4>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:/O2>
    )
    if(FROM_NATIVE_ARCH)
        target_compile_options(from PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
            $<$<COMPILE_LANGUAGE:C>:/arch:AVX2>
        )
    endif()
    target_link_libraries(from PRIVATE d3d11 dxgi d3dcompiler)
else()
    target_compile_options(from PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wno-unused-parameter>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3;-ffast-math>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:-O3;-ffast-math>
    )
    if(FROM_NATIVE_ARCH)
        target_compile_options(from PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:-march=native>
        )
    endif()
    target_compile_options(from PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g;-fsanitize=address,undefined>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:C>>:-O0;-g;-fsanitize=address,undefined>
    )
    target_link_options(from PRIVATE "$<$<CONFIG:Debug>:-fsanitize=address,undefined>")
    target_link_libraries(from PRIVATE m)
endif()

find_package(Threads REQUIRED)
target_link_libraries(from PRIVATE Threads::Threads)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(from PRIVATE OpenMP::OpenMP_CXX)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(from PRIVATE OpenMP::OpenMP_C)
endif()

