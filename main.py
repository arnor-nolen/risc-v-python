import glob
import struct
import numpy as np
from elftools.elf.elffile import ELFFile
from enum import Enum, auto
import csv


class Op(Enum):
    LUI = auto()
    AUIPC = auto()
    JAL = auto()
    JALR = auto()
    BEQ = auto()
    BNE = auto()
    BLT = auto()
    BGE = auto()
    BLTU = auto()
    BGEU = auto()
    LB = auto()
    LH = auto()
    LW = auto()
    LBU = auto()
    LHU = auto()
    SB = auto()
    SH = auto()
    SW = auto()
    ADDI = auto()
    SLTI = auto()
    SLTIU = auto()
    XORI = auto()
    ORI = auto()
    ANDI = auto()
    SRLI = auto()
    SRAI = auto()
    ADD = auto()
    SUB = auto()
    SLL = auto()
    SLT = auto()
    SLTU = auto()
    XOR = auto()
    SRL = auto()
    SRA = auto()
    OR = auto()
    AND = auto()
    FENCE = auto()
    ECALL = auto()
    EBREAK = auto()


class InsType(Enum):
    R = auto()
    I = auto()
    S = auto()
    B = auto()
    U = auto()
    J = auto()


# type_mapping = {
#     Op.LUI: InsType.U,
#     Op.AUIPC: InsType.U,
#     Op.JAL: InsType.J,
#     Op.JALR: InsType.I,
#     Op.BEQ: InsType.B,
#     Op.BNE: InsType.B,
#     Op.BLT: InsType.B,
#     Op.BGE: InsType.B,
#     Op.BLTU: InsType.B,
#     Op.BGEU: InsType.B,
#     Op.LB: InsType.U,
#     Op.LH: InsType.U,
#     Op.LW: InsType.U,
#     Op.LBU: InsType.U,
#     Op.LHU: InsType.U,
#     Op.SB: InsType.S,
#     Op.SH: InsType.S,
#     Op.SW: InsType.S,
#     Op.ADDI: InsType.I,
#     Op.SLTI: InsType.I,
#     Op.SLTIU: InsType.I,
#     Op.XORI: InsType.I,
#     Op.ORI: InsType.I,
#     Op.ANDI: InsType.I,
#     Op.SLLI: InsType.I,
#     Op.SRLI: InsType.I,
#     Op.SRAI: InsType.I,
#     Op.ADD: InsType.R,
#     Op.SUB: InsType.R,
#     Op.SLL: InsType.R,
#     Op.SLT: InsType.R,
#     Op.SLTU: InsType.R,
#     Op.XOR: InsType.R,
#     Op.SRL: InsType.R,
#     Op.SRA: InsType.R,
#     Op.OR: InsType.R,
#     Op.AND: InsType.R,
#     Op.FENCE: InsType.U,
#     Op.ECALL: InsType.U,
#     Op.EBREAK: InsType.U,
# }


class Ins:
    def __init__(self, op: Op):

        pass


# class Op:
#     def __init__(self, name, format_str):
#         self.name = name
#         self.format_str = format_str

#     def __repr__(self):
#         return f'{self.name:8} {self.format_str}'


def get_mask(start, end):
    return ((1 << (start + 1)) - 1) - ((1 << end) - 1)


def get_bits(value, start, end):
    return (value & get_mask(start, end)) >> end


def step():
    pass


# def setup():
# with open('./isa.csv', 'r') as file:
#     csv_file = [x for x in csv.reader(file, delimiter=' ')]
# Op = Enum('Op', ' '.join([x[-1] for x in csv_file]))
# print(Op.BEQ)

# for row in csv.reader(file, delimiter=' '):
#     operand = Op(row[-1], ' '.join(row[0:-2]))
#     ops.append(operand)
#     print(operand)


def execute_elf(file):
    # setup()

    elf = ELFFile(file)
    text_init = elf.get_section_by_name('.text.init')
    start_addr = np.uint32(text_init.header['sh_addr'])
    data = text_init.data()

    size = text_init.header['sh_size']

    registers = np.zeros(32, dtype=np.uint32)
    pc = start_addr

    def dump_regs():
        for i in range(32):
            print(f'x{i:02}={registers[i]:032b}')

    offset = 0
    while offset + 4 <= size:
        ins = struct.unpack('I', data[offset : offset + 4])[0]
        opcode = get_bits(ins, 6, 0)

        if opcode == 0b0110111:
            # LUI instruction
            # U type
            imm = get_bits(ins, 31, 12) << 12
            rd = get_bits(ins, 11, 7)
            print(f'{pc:08x} {ins:08x} {opcode=:07b} LUI {imm=:08x} {rd=}')

            registers[rd] = imm

        elif opcode == 0b0010111:
            # AUIPC instruction
            # U type
            imm = get_bits(ins, 31, 12) << 12
            rd = get_bits(ins, 11, 7)
            print(f'{pc:08x} {ins:08x} {opcode=:07b} AUIPC {imm=:08x} {rd=}')

            registers[rd] = pc + imm

        elif opcode == 0b1101111:
            # JAL instruction
            # J type
            imm = (
                (get_bits(ins, 31, 31) << 20)
                + (get_bits(ins, 30, 21) << 1)
                + (get_bits(ins, 20, 20) << 11)
                + (get_bits(ins, 19, 12) << 12)
            )
            rd = get_bits(ins, 11, 7)
            print(f'{pc:08x} {ins:08x} {opcode=:07b} JAL {imm=:08x} {rd=}')

            # Sign extending the immediate
            if imm & (1 << 20):
                # negative number
                imm = ((imm ^ ((1 << 20) - 1)) + 1) & ((1 << 20) - 1)
                pc = pc - imm - 4
            else:
                # positive number
                pc = pc + imm - 4

            registers[rd] = pc + 4

        else:
            print(f'{pc:08x} {ins:08x} {opcode=:07b} UNKNOWN')

        # dump_regs()

        # Normal flow, go to the next instruction
        pc += 4
        offset = pc - start_addr


if __name__ == '__main__':
    paths = [
        x
        for x in glob.glob('./riscv-tests/isa/rv32ui-p-jal')
        if len(x.split('.')) == 2
    ]
    for path in paths:
        print(f'Executing file: {path}')
        with open(path, 'rb') as file:
            execute_elf(file)
            # break
