# Stage 1d.4+5: trace-parser now embeds the MICRO 2025 traces_enhanced data
# classes + generated protobuf bindings directly (rather than consuming them
# from util/traces_enhanced's Makefile output).  This keeps the CMake build
# self-contained and matches MICRO 2025's layering.

FILE(GLOB files *.cc)
FILE(GLOB enhanced_src_files $ENV{ACCELSIM_ROOT}/../util/traces_enhanced/src/*.cc)
FILE(GLOB pb_gen_src_files   $ENV{ACCELSIM_ROOT}/../util/traces_enhanced/pb_trace/src/*.pb.cc)

include_directories($ENV{CUDA_INSTALL_PATH}/include)
include_directories($ENV{GPGPUSIM_ROOT}/libcuda)
include_directories($ENV{GPGPUSIM_ROOT}/src)
include_directories($ENV{ACCELSIM_ROOT}/../util/traces_enhanced/src)
include_directories($ENV{ACCELSIM_ROOT}/../util/traces_enhanced/pb_trace/include)

add_library(trace-parser STATIC
    ${files}
    ${enhanced_src_files}
    ${pb_gen_src_files}
)

# PUBLIC so that consumers (trace-driven, accel-sim.out) pick up -lprotobuf
target_link_libraries(trace-parser PUBLIC protobuf)
