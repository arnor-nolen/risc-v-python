import glob
import struct
import numpy as np
from elftools.elf.elffile import ELFFile
from enum import Enum, auto
import csv


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


def parse_elf(file):
    elf = ELFFile(file)
    text_init = elf.get_section_by_name('.text.init')
    start_addr = np.uint32(text_init.header['sh_addr'])
    data = text_init.data()
    size = text_init.header['sh_size']
    return data, size, start_addr


def execute_elf(file):
    data, size, start_addr = parse_elf(file)
    return execute(data, size, start_addr)


def execute(data, size, start_addr):
    registers = np.zeros(32, dtype=np.uint32)
    pc = start_addr

    def dump_regs():
        for i in range(32):
            print(f'x{i:02}={registers[i]:032b}')

    offset = 0
    while offset + 4 <= size:
        ins = struct.unpack('I', data[offset : offset + 4])[0]
        opcode = Opcode(get_bits(ins, 6, 0))

        print(f'{pc:08x} {ins:08x} {opcode.name} ', end='')

        if opcode == Opcode.LUI:
            # U type
            imm = get_bits(ins, 31, 12) << 12
            rd = get_bits(ins, 11, 7)
            print(f'{imm=:08x} {rd=}')

            registers[rd] = imm

        elif opcode == Opcode.AUIPC:
            # U type
            imm = get_bits(ins, 31, 12) << 12
            rd = get_bits(ins, 11, 7)
            print(f'{imm=:08x} {rd=}')

            registers[rd] = pc + imm

        elif opcode == Opcode.JAL:
            # J type
            imm = (
                (get_bits(ins, 31, 31) << 20)
                + (get_bits(ins, 30, 21) << 1)
                + (get_bits(ins, 20, 20) << 11)
                + (get_bits(ins, 19, 12) << 12)
            )
            rd = get_bits(ins, 11, 7)
            print(f'{imm=:08x} {rd=}')

            # Sign extending the immediate
            if imm & (1 << 19):
                # negative number
                imm = ((imm ^ ((1 << 20) - 1)) + 1) & ((1 << 20) - 1)
                branch_to = pc - imm - 4
            else:
                # positive number
                branch_to = pc + imm - 4

            registers[rd] = pc + 4
            pc = branch_to

        elif opcode == Opcode.JALR:
            # I type
            imm = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)
            print(f'{imm=:08x} {rs1=} {funct3=} {rd=}')

            # TODO: Let's hope it works, need to test
            # Sign extending the immediate
            if imm & (1 << 11):
                # negative number
                imm = ((imm ^ ((1 << 12) - 1)) + 1) & ((1 << 12) - 1)
                branch_to = (registers[rs1] - imm - 4) & ((1 << 32) - 2)
            else:
                # positive number
                branch_to = (registers[rs1] + imm - 4) & ((1 << 32) - 2)

            registers[rd] = pc + 4
            pc = branch_to

        elif opcode == Opcode.BRANCH:
            # BEQ, BNE, BLT, BGE, BLTU, BGEU instructions
            # B type
            imm = (
                (get_bits(ins, 31, 31) << 20)
                + (get_bits(ins, 30, 25) << 5)
                + (get_bits(ins, 11, 8) << 1)
                + (get_bits(ins, 7, 7) << 11)
            )
            rs2 = get_bits(ins, 24, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)

            # TODO: Let's hope it works, need to test
            # Sign extending the immediate
            if imm & (1 << 11):
                # negative number
                imm = ((imm ^ ((1 << 12) - 1)) + 1) & ((1 << 12) - 1)
                branch_to = pc - imm - 4
            else:
                # positive number
                branch_to = pc + imm - 4

            def print_branch(ins_name):
                print(f'{ins_name} {imm=:08x} {rs1=} {rs2=} {funct3=} {rd=}')

            if funct3 == 0b000:
                # BEQ instruction
                print_branch('BEQ')
                if registers[rs1] == registers[rs2]:
                    pc = branch_to
            elif funct3 == 0b001:
                # BNE instruction
                print_branch('BNE')
                if registers[rs1] != registers[rs2]:
                    pc = branch_to
            elif funct3 == 0b100:
                # BLT instruction
                print_branch('BLT')
                # Check if values are negative
                value1 = registers[rs1]
                if value1 & (1 << 31):
                    value1 = ((value1 ^ ((1 << 32) - 1)) + 1) & ((1 << 32) - 1)
                value2 = registers[rs2]
                if value2 & (1 << 31):
                    value2 = ((value2 ^ ((1 << 32) - 1)) + 1) & ((1 << 32) - 1)
                if value1 < value2:
                    pc = branch_to
            elif funct3 == 0b101:
                # BGE instruction
                print_branch('BGE')
                # Check if values are negative
                value1 = registers[rs1]
                if value1 & (1 << 31):
                    value1 = ((value1 ^ ((1 << 32) - 1)) + 1) & ((1 << 32) - 1)
                value2 = registers[rs2]
                if value2 & (1 << 31):
                    value2 = ((value2 ^ ((1 << 32) - 1)) + 1) & ((1 << 32) - 1)
                if value1 > value2:
                    pc = branch_to
            elif funct3 == 0b110:
                # BLTU instruction
                print_branch('BLTU')
                if registers[rs1] < registers[rs2]:
                    pc = branch_to
            elif funct3 == 0b111:
                # BGEU instruction
                print_branch('BGEU')
                if registers[rs1] > registers[rs2]:
                    pc = branch_to
            else:
                # No instruction with such funct3
                print_branch('UNKNOWN')
                raise Exception(
                    f'Wrong {funct3=:03b} for branch instructions!'
                )

        elif opcode == Opcode.LOAD:
            # LB, LH, LW, LBU, LHU instructions
            # I type
            imm = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)
            print(f'{imm=:08x} {rs1=} {funct3=} {rd=}')

        elif opcode == Opcode.STORE:
            # SB, SH, SW instructions
            # S type
            print(f'')
        elif opcode == Opcode.OP:
            # Arithmetic instrutions (10 pcs)
            print(f'')
        elif opcode == Opcode.OP_IMM:
            # Arithmetic instrutions with immediate (9 pcs)
            print(f'')
        elif opcode == Opcode.MISC_MEM:
            # FENCE instruction
            print(f'')
        elif opcode == Opcode.SYSTEM:
            # ECALL, EBREAK instructions
            print(f'')

        else:
            print(f'UNKNOWN')

        # dump_regs()

        # Normal flow, go to the next instruction
        pc += 4
        offset = pc - start_addr


if __name__ == '__main__':
    paths = [
        x
        for x in glob.glob('./riscv-tests/isa/rv32ui-p-*')
        if len(x.split('.')) == 2
    ]
    for path in paths:
        print(f'Executing file: {path}')
        with open(path, 'rb') as file:
            execute_elf(file)
            break
