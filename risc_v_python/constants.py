from enum import Enum

# Values for all Enums are taken from RISC-V reference manual
class Opcode(Enum):
    LUI = 0b0110111
    AUIPC = 0b0010111
    JAL = 0b1101111
    JALR = 0b1100111
    BRANCH = 0b1100011
    LOAD = 0b0000011
    STORE = 0b0100011
    OP = 0b0110011
    OP_IMM = 0b0010011
    MISC_MEM = 0b0001111
    SYSTEM = 0b1110011


class BranchOp(Enum):
    BEQ = 0b000
    BNE = 0b001
    BLT = 0b100
    BGE = 0b101
    BLTU = 0b110
    BGEU = 0b111


class LoadOp(Enum):
    LB = 0b000
    LH = 0b001
    LW = 0b010
    LBU = 0b100
    LHU = 0b101


class StoreOp(Enum):
    SB = 0b000
    SH = 0b001
    SW = 0b010


class OpImm(Enum):
    ADDI = 0b000
    SLTI = 0b010
    SLTIU = 0b011
    XORI = 0b100
    ORI = 0b110
    ANDI = 0b111
    SLLI = 0b001
    SRLI_SRAI = 0b101


class Op(Enum):
    ADD_SUB = 0b000
    SLL = 0b001
    SLT = 0b010
    SLTU = 0b011
    XOR = 0b100
    SRL_SRA = 0b101
    OR = 0b110
    AND = 0b111


class SystemOp(Enum):
    ECALL = 0b000000000000
    EBREAK = 0b000000000001


class CsrOp(Enum):
    CSRRW = 0b001
    CSRRS = 0b010
    CSRRC = 0b011
    CSRRWI = 0b101
    CSRRSI = 0b110
    CSRRCI = 0b111
