// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu
// Phase 3 Step A: opcode→op_type remapping aligned with MICRO 2025
// Huerta et al. (turing_opcode.h from their SM75_RTX2070_S config).

#ifndef TURING_OPCODE_H
#define TURING_OPCODE_H

#include <string>
#include <unordered_map>
#include "abstract_hardware_model.h"
#include "trace_opcode.h"

#define TURING_BINART_VERSION 75

static const std::unordered_map<std::string, OpcodeChar> Turing_OpcodeMap = {
    // Floating Point 32 Instructions
    {"FADD", OpcodeChar(OP_FADD, SP_OP)},
    {"FADD32I", OpcodeChar(OP_FADD32I, SP_OP)},
    {"FCHK", OpcodeChar(OP_FCHK, SP_OP)},
    {"FFMA32I", OpcodeChar(OP_FFMA32I, SP_OP)},
    {"FFMA", OpcodeChar(OP_FFMA, SP_OP)},
    {"FMNMX", OpcodeChar(OP_FMNMX, SP_OP)},
    {"FMUL", OpcodeChar(OP_FMUL, SP_OP)},
    {"FMUL32I", OpcodeChar(OP_FMUL32I, SP_OP)},
    {"FSEL", OpcodeChar(OP_FSEL, SP_OP)},
    {"FSET", OpcodeChar(OP_FSET, SP_OP)},
    {"FSETP", OpcodeChar(OP_FSETP, SP_OP)},
    {"FSWZADD", OpcodeChar(OP_FSWZADD, SP_OP)},
    // SFU
    {"MUFU", OpcodeChar(OP_MUFU, SFU_OP)},

    // Floating Point 16 Instructions [MICRO25: SP_OP → HALF_OP]
    {"HADD2", OpcodeChar(OP_HADD2, HALF_OP)},
    {"HADD2_32I", OpcodeChar(OP_HADD2_32I, HALF_OP)},
    {"HFMA2", OpcodeChar(OP_HFMA2, HALF_OP)},
    {"HFMA2_32I", OpcodeChar(OP_HFMA2_32I, HALF_OP)},
    {"HMUL2", OpcodeChar(OP_HMUL2, HALF_OP)},
    {"HMUL2_32I", OpcodeChar(OP_HMUL2_32I, HALF_OP)},
    {"HSET2", OpcodeChar(OP_HSET2, HALF_OP)},
    {"HSETP2", OpcodeChar(OP_HSETP2, HALF_OP)},

    // Tensor Core Instructions [MICRO25: SPECIALIZED_UNIT_3 → TENSOR_CORE_OP]
    {"HMMA", OpcodeChar(OP_HMMA, TENSOR_CORE_OP)},
    {"BMMA", OpcodeChar(OP_BMMA, TENSOR_CORE_OP)},
    {"IMMA", OpcodeChar(OP_IMMA, TENSOR_CORE_OP)},

    // Double Point Instructions
    {"DADD", OpcodeChar(OP_DADD, DP_OP)},
    {"DFMA", OpcodeChar(OP_DFMA, DP_OP)},
    {"DMUL", OpcodeChar(OP_DMUL, DP_OP)},
    {"DSETP", OpcodeChar(OP_DSETP, DP_OP)},

    // Integer Instructions
    {"BMSK", OpcodeChar(OP_BMSK, INTP_OP)},
    {"BREV", OpcodeChar(OP_BREV, INTP_OP)},
    {"FLO", OpcodeChar(OP_FLO, INTP_OP)},
    {"IABS", OpcodeChar(OP_IABS, INTP_OP)},
    {"IADD", OpcodeChar(OP_IADD, INTP_OP)},
    {"IADD3", OpcodeChar(OP_IADD3, INTP_OP)},
    {"IADD32I", OpcodeChar(OP_IADD32I, INTP_OP)},
    {"IDP", OpcodeChar(OP_IDP, INTP_OP)},
    {"IDP4A", OpcodeChar(OP_IDP4A, INTP_OP)},
    // [MICRO25: IMAD goes to SP_OP, not INTP_OP — shares FP32 FU]
    {"IMAD", OpcodeChar(OP_IMAD, SP_OP)},
    {"IMNMX", OpcodeChar(OP_IMNMX, INTP_OP)},
    {"IMUL", OpcodeChar(OP_IMUL, INTP_OP)},
    {"IMUL32I", OpcodeChar(OP_IMUL32I, INTP_OP)},
    {"ISCADD", OpcodeChar(OP_ISCADD, INTP_OP)},
    {"ISCADD32I", OpcodeChar(OP_ISCADD32I, INTP_OP)},
    {"ISETP", OpcodeChar(OP_ISETP, INTP_OP)},
    {"LEA", OpcodeChar(OP_LEA, INTP_OP)},
    {"LOP", OpcodeChar(OP_LOP, INTP_OP)},
    {"LOP3", OpcodeChar(OP_LOP3, INTP_OP)},
    {"LOP32I", OpcodeChar(OP_LOP32I, INTP_OP)},
    {"POPC", OpcodeChar(OP_POPC, INTP_OP)},
    {"SHF", OpcodeChar(OP_SHF, INTP_OP)},
    {"SHL", OpcodeChar(OP_SHL, INTP_OP)},
    {"SHR", OpcodeChar(OP_SHR, INTP_OP)},
    {"VABSDIFF", OpcodeChar(OP_VABSDIFF, INTP_OP)},
    {"VABSDIFF4", OpcodeChar(OP_VABSDIFF4, INTP_OP)},

    // Conversion Instructions [MICRO25: ALU_OP → SFU_OP/DP_OP]
    {"F2F", OpcodeChar(OP_F2F, DP_OP)},
    {"F2FP", OpcodeChar(OP_F2FP, SFU_OP)},
    {"F2I", OpcodeChar(OP_F2I, SFU_OP)},
    {"I2F", OpcodeChar(OP_I2F, SFU_OP)},
    {"I2I", OpcodeChar(OP_I2I, SFU_OP)},
    {"I2IP", OpcodeChar(OP_I2IP, SFU_OP)},
    {"FRND", OpcodeChar(OP_FRND, SFU_OP)},

    // Movement Instructions
    {"MOV", OpcodeChar(OP_MOV, ALU_OP)},
    {"MOV32I", OpcodeChar(OP_MOV32I, ALU_OP)},
    {"MOVM", OpcodeChar(OP_MOVM, ALU_OP)},
    {"PRMT", OpcodeChar(OP_PRMT, ALU_OP)},
    {"SEL", OpcodeChar(OP_SEL, ALU_OP)},
    {"SGXT", OpcodeChar(OP_SGXT, ALU_OP)},
    {"SHFL", OpcodeChar(OP_SHFL, ALU_OP)},

    // Predicate Instructions [MICRO25: ALU_OP → PREDICATE_OP]
    {"PLOP3", OpcodeChar(OP_PLOP3, PREDICATE_OP)},
    {"PSETP", OpcodeChar(OP_PSETP, PREDICATE_OP)},
    {"P2R", OpcodeChar(OP_P2R, PREDICATE_OP)},
    {"R2P", OpcodeChar(OP_R2P, PREDICATE_OP)},

    // Load/Store Instructions
    {"LD", OpcodeChar(OP_LD, LOAD_OP)},
    {"LDC", OpcodeChar(OP_LDC, ALU_OP)},
    {"LDG", OpcodeChar(OP_LDG, LOAD_OP)},
    {"LDL", OpcodeChar(OP_LDL, LOAD_OP)},
    {"LDS", OpcodeChar(OP_LDS, LOAD_OP)},
    {"LDSM", OpcodeChar(OP_LDSM, LOAD_OP)},
    {"ST", OpcodeChar(OP_ST, STORE_OP)},
    {"STG", OpcodeChar(OP_STG, STORE_OP)},
    {"STL", OpcodeChar(OP_STL, STORE_OP)},
    {"STS", OpcodeChar(OP_STS, STORE_OP)},
    {"MATCH", OpcodeChar(OP_MATCH, ALU_OP)},
    {"QSPC", OpcodeChar(OP_QSPC, ALU_OP)},
    {"ATOM", OpcodeChar(OP_ATOM, STORE_OP)},
    {"ATOMS", OpcodeChar(OP_ATOMS, STORE_OP)},
    {"ATOMG", OpcodeChar(OP_ATOMG, STORE_OP)},
    {"RED", OpcodeChar(OP_RED, STORE_OP)},
    {"CCTL", OpcodeChar(OP_CCTL, ALU_OP)},
    {"CCTLL", OpcodeChar(OP_CCTLL, ALU_OP)},
    {"ERRBAR", OpcodeChar(OP_ERRBAR, ALU_OP)},
    {"MEMBAR", OpcodeChar(OP_MEMBAR, MEMORY_BARRIER_OP)},
    {"CCTLT", OpcodeChar(OP_CCTLT, ALU_OP)},

    // Uniform Datapath Instructions [MICRO25: SPECIALIZED_UNIT_4 → UNIFORM_OP]
    {"R2UR", OpcodeChar(OP_R2UR, UNIFORM_OP)},
    {"S2UR", OpcodeChar(OP_S2UR, UNIFORM_OP)},
    {"UBMSK", OpcodeChar(OP_UBMSK, UNIFORM_OP)},
    {"UBREV", OpcodeChar(OP_UBREV, UNIFORM_OP)},
    {"UCLEA", OpcodeChar(OP_UCLEA, UNIFORM_OP)},
    {"UFLO", OpcodeChar(OP_UFLO, UNIFORM_OP)},
    {"UIADD3", OpcodeChar(OP_UIADD3, UNIFORM_OP)},
    {"UIMAD", OpcodeChar(OP_UIMAD, UNIFORM_OP)},
    {"UISETP", OpcodeChar(OP_UISETP, UNIFORM_OP)},
    {"ULDC", OpcodeChar(OP_ULDC, UNIFORM_OP)},
    {"ULEA", OpcodeChar(OP_ULEA, UNIFORM_OP)},
    {"ULOP", OpcodeChar(OP_ULOP, UNIFORM_OP)},
    {"ULOP3", OpcodeChar(OP_ULOP3, UNIFORM_OP)},
    {"ULOP32I", OpcodeChar(OP_ULOP32I, UNIFORM_OP)},
    {"UMOV", OpcodeChar(OP_UMOV, UNIFORM_OP)},
    {"UP2UR", OpcodeChar(OP_UP2UR, UNIFORM_OP)},
    {"UPLOP3", OpcodeChar(OP_UPLOP3, UNIFORM_OP)},
    {"UPOPC", OpcodeChar(OP_UPOPC, UNIFORM_OP)},
    {"UPRMT", OpcodeChar(OP_UPRMT, UNIFORM_OP)},
    {"UPSETP", OpcodeChar(OP_UPSETP, UNIFORM_OP)},
    {"UR2UP", OpcodeChar(OP_UR2UP, UNIFORM_OP)},
    {"USEL", OpcodeChar(OP_USEL, UNIFORM_OP)},
    {"USGXT", OpcodeChar(OP_USGXT, UNIFORM_OP)},
    {"USHF", OpcodeChar(OP_USHF, UNIFORM_OP)},
    {"USHL", OpcodeChar(OP_USHL, UNIFORM_OP)},
    {"USHR", OpcodeChar(OP_USHR, UNIFORM_OP)},
    {"VOTEU", OpcodeChar(OP_VOTEU, UNIFORM_OP)},

    // Texture Instructions (keep as SPECIALIZED_UNIT_2)
    {"TEX", OpcodeChar(OP_TEX, SPECIALIZED_UNIT_2_OP)},
    {"TLD", OpcodeChar(OP_TLD, SPECIALIZED_UNIT_2_OP)},
    {"TLD4", OpcodeChar(OP_TLD4, SPECIALIZED_UNIT_2_OP)},
    {"TMML", OpcodeChar(OP_TMML, SPECIALIZED_UNIT_2_OP)},
    {"TXD", OpcodeChar(OP_TXD, SPECIALIZED_UNIT_2_OP)},
    {"TXQ", OpcodeChar(OP_TXQ, SPECIALIZED_UNIT_2_OP)},

    // Surface Instructions
    {"SUATOM", OpcodeChar(OP_SUATOM, ALU_OP)},
    {"SULD", OpcodeChar(OP_SULD, ALU_OP)},
    {"SURED", OpcodeChar(OP_SURED, ALU_OP)},
    {"SUST", OpcodeChar(OP_SUST, ALU_OP)},

    // Control Instructions [MICRO25: SPECIALIZED_UNIT_1 → BRANCH_OP]
    {"BMOV", OpcodeChar(OP_BMOV, BRANCH_OP)},
    {"BPT", OpcodeChar(OP_BPT, BRANCH_OP)},
    {"BRA", OpcodeChar(OP_BRA, BRANCH_OP)},
    {"BREAK", OpcodeChar(OP_BREAK, BRANCH_OP)},
    {"BRX", OpcodeChar(OP_BRX, BRANCH_OP)},
    {"BRXU", OpcodeChar(OP_BRXU, BRANCH_OP)},
    {"BSSY", OpcodeChar(OP_BSSY, BRANCH_OP)},
    {"BSYNC", OpcodeChar(OP_BSYNC, BRANCH_OP)},
    {"CALL", OpcodeChar(OP_CALL, CALL_OPS)},
    {"EXIT", OpcodeChar(OP_EXIT, EXIT_OPS)},
    {"JMP", OpcodeChar(OP_JMP, BRANCH_OP)},
    {"JMX", OpcodeChar(OP_JMX, BRANCH_OP)},
    {"JMXU", OpcodeChar(OP_JMXU, BRANCH_OP)},
    {"KILL", OpcodeChar(OP_KILL, MISCELLANEOUS_NO_QUEUE_OP)},
    {"NANOSLEEP", OpcodeChar(OP_NANOSLEEP, MISCELLANEOUS_NO_QUEUE_OP)},
    {"RET", OpcodeChar(OP_RET, RET_OPS)},
    {"RPCMOV", OpcodeChar(OP_RPCMOV, BRANCH_OP)},
    {"RTT", OpcodeChar(OP_RTT, BRANCH_OP)},
    {"WARPSYNC", OpcodeChar(OP_WARPSYNC, BRANCH_OP)},
    {"YIELD", OpcodeChar(OP_YIELD, BRANCH_OP)},

    // Miscellaneous Instructions [MICRO25: ALU_OP → MISCELLANEOUS_NO_QUEUE_OP]
    {"B2R", OpcodeChar(OP_B2R, MISCELLANEOUS_NO_QUEUE_OP)},
    {"BAR", OpcodeChar(OP_BAR, BARRIER_OP)},
    {"CS2R", OpcodeChar(OP_CS2R, INTP_OP)},
    {"CSMTEST", OpcodeChar(OP_CSMTEST, MISCELLANEOUS_NO_QUEUE_OP)},
    {"DEPBAR", OpcodeChar(OP_DEPBAR, ALU_OP)},
    {"GETLMEMBASE", OpcodeChar(OP_GETLMEMBASE, MISCELLANEOUS_NO_QUEUE_OP)},
    {"LEPC", OpcodeChar(OP_LEPC, MISCELLANEOUS_NO_QUEUE_OP)},
    {"NOP", OpcodeChar(OP_NOP, MISCELLANEOUS_NO_QUEUE_OP)},
    {"PMTRIG", OpcodeChar(OP_PMTRIG, MISCELLANEOUS_NO_QUEUE_OP)},
    {"R2B", OpcodeChar(OP_R2B, MISCELLANEOUS_NO_QUEUE_OP)},
    {"S2R", OpcodeChar(OP_S2R, MISCELLANEOUS_QUEUE_OP)},
    {"SETCTAID", OpcodeChar(OP_SETCTAID, MISCELLANEOUS_NO_QUEUE_OP)},
    {"SETLMEMBASE", OpcodeChar(OP_SETLMEMBASE, MISCELLANEOUS_NO_QUEUE_OP)},
    {"VOTE", OpcodeChar(OP_VOTE, MISCELLANEOUS_NO_QUEUE_OP)},
    {"VOTE_VTG", OpcodeChar(OP_VOTE_VTG, MISCELLANEOUS_NO_QUEUE_OP)},

};

#endif
