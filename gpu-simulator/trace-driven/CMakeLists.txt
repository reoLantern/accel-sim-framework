# Stage 1d.4+5: trace-driven includes traces_enhanced headers (for
# traced_instruction, string_utilities, traced_execution) via relative
# paths; it does not compile any traces_enhanced .cc — those live in
# trace-parser's library.  We still need the -I path for direct
# trace-driven include sites (sm_75/sm_90 opcode tables are resolved
# through the MICRO 2025 OpcodeMap, kept in ISA_Def).

FILE(GLOB files *.cc)

include_directories($ENV{CUDA_INSTALL_PATH}/include)
include_directories($ENV{GPGPUSIM_ROOT}/libcuda)
include_directories($ENV{GPGPUSIM_ROOT}/src)
include_directories($ENV{ACCELSIM_ROOT}/../util/traces_enhanced/src)
include_directories($ENV{ACCELSIM_ROOT}/../util/traces_enhanced/pb_trace/include)

add_library(trace-driven STATIC ${files})
target_link_libraries(trace-driven PUBLIC trace-parser)
