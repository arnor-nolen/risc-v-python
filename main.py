import glob
import struct
import numpy as np
from elftools.elf.elffile import ELFFile
from enum import Enum


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


# class InsType(Enum):
#     R = auto()
#     I = auto()
#     S = auto()
#     B = auto()
#     U = auto()
#     J = auto()


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


def get_mask(start, end):
    return ((1 << (start + 1)) - 1) - ((1 << end) - 1)


def get_bits(value, start, end):
    return (value & get_mask(start, end)) >> end


def sign_extend(value, cur_size, new_size):
    if value & get_mask(cur_size - 1, cur_size - 1):
        # If negative, sign extend
        value = get_mask(new_size - 1, cur_size) | value
    return value


def step():
    pass


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
    # Allocate 4 KB heap
    memory = np.zeros(4 * 1024, dtype=np.uint8)
    pc = start_addr

    def dump_regs():
        for i in range(32):
            print(f'x{i:02}={registers[i]:032b}')

    offset = 0
    while offset + 4 <= size:
        ins = struct.unpack('I', data[offset : offset + 4])[0]
        opcode = Opcode(get_bits(ins, 6, 0))

        print(f'{pc:08x} {ins:08x} {opcode.name.rjust(6)} ', end='')

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

            pc = pc + imm - 4
            registers[rd] = pc + 4

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
            imm = sign_extend(imm, 32, 32)
            registers[rd] = pc + 4
            pc = pc + imm - 4

        elif opcode == Opcode.JALR:
            # I type
            imm = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)
            print(f'{imm=:08x} {rs1=} {funct3=} {rd=}')

            # TODO: Let's hope it works, need to test
            # Sign extending the immediate
            imm = sign_extend(imm, 12, 32)

            branch_to = ((registers[rs1] + imm) & get_mask(31, 1)) - 4
            registers[rd] = pc + 4
            pc = branch_to

        # Partially UNIMPLEMENTED
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
            imm = sign_extend(imm, 12, 32)

            branch_to = pc + imm - 4

            branch_op = BranchOp(funct3)
            print(f'{branch_op.name.rjust(4)} {imm=:08x} {rs1=} {rs2=} {rd=}')

            if branch_op == BranchOp.BEQ:
                if registers[rs1] == registers[rs2]:
                    pc = branch_to
            elif branch_op == BranchOp.BNE:
                if registers[rs1] != registers[rs2]:
                    pc = branch_to
            elif branch_op == BranchOp.BLT:
                raise Exception("Unimplemented!")
            elif branch_op == BranchOp.BGE:
                raise Exception("Unimplemented!")
            elif branch_op == BranchOp.BLTU:
                if registers[rs1] < registers[rs2]:
                    pc = branch_to
            elif branch_op == BranchOp.BGEU:
                if registers[rs1] > registers[rs2]:
                    pc = branch_to
            else:
                # No instruction with such branch_op
                raise Exception(
                    f'Wrong {branch_op=:03b} for branch instructions!'
                )

        # UNIMPLEMENTED
        elif opcode == Opcode.LOAD:
            # LB, LH, LW, LBU, LHU instructions
            # I type
            imm = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)
            load_op = LoadOp(funct3)

            print(f'{load_op} {imm=:08x} {rs1=} {rd=}')

        # UNIMPLEMENTED
        elif opcode == Opcode.STORE:
            # SB, SH, SW instructions
            # S type
            print(f'')

        # Partially UNIMPLEMENTED
        elif opcode == Opcode.OP_IMM:
            # Arithmetic instrutions with immediate (9 pcs)
            imm = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)

            op_imm = OpImm(funct3)

            print(f'{op_imm.name.rjust(9)} {imm=:08x} {rs1=} {rd=}')
            if op_imm == OpImm.ADDI:
                imm = sign_extend(imm, 12, 32)
                registers[rd] = registers[rs1] + imm
            elif op_imm == OpImm.SLTI:
                imm = sign_extend(imm, 12, 32)
                registers[rd] = int(registers[rs1] < imm)
            elif op_imm == OpImm.SLTIU:
                imm = sign_extend(imm, 12, 32)
                raise Exception("Unimplemented!")
            elif op_imm == OpImm.XORI:
                imm = sign_extend(imm, 12, 32)
                registers[rd] = registers[rs1] ^ imm
            elif op_imm == OpImm.ORI:
                imm = sign_extend(imm, 12, 32)
                registers[rd] = registers[rs1] | imm
            elif op_imm == OpImm.ANDI:
                imm = sign_extend(imm, 12, 32)
                registers[rd] = registers[rs1] & imm
            elif op_imm == OpImm.SLLI:
                shamt = get_bits(imm, 4, 0)
                registers[rd] = registers[rs1] << shamt
            elif op_imm == OpImm.SRLI_SRAI:
                shamt = get_bits(imm, 4, 0)
                funct7 = get_bits(imm, 11, 5)
                if funct7 == 0b0000000:
                    # SRLI instruction
                    registers[rd] = registers[rs1] >> shamt
                elif funct7 == 0b0100000:
                    # SRAI instruction
                    registers[rd] = sign_extend(
                        registers[rs1] >> shamt, 32 - shamt, 32
                    )
                else:
                    raise Exception("Unknown funct7 for SRLI_SRAI!")

        # Partially UNIMPLEMENTED
        elif opcode == Opcode.OP:
            # Arithmetic instrutions (10 pcs)
            # Arithmetic instrutions with immediate (9 pcs)
            funct7 = get_bits(ins, 31, 25)
            rs2 = get_bits(ins, 24, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)

            op = Op(funct3)

            print(f'{op.name.rjust(7)} {funct7=} {rs1=} {rs2=} {rd=}')

            if op == Op.ADD_SUB:
                if funct7 == 0b0000000:
                    # ADD instruction
                    registers[rd] = registers[rs1] + registers[rs2]
                elif funct7 == 0b0100000:
                    # SUB instruction
                    registers[rd] = registers[rs1] - registers[rs2]
            elif op == Op.SLL:
                shamt = get_bits(registers[rs2], 4, 0)
                registers[rd] = registers[rs1] << shamt
            elif op == Op.SLT:
                registers[rd] = int(registers[rs1] < registers[rs2])
            elif op == Op.SLTU:
                raise Exception("Unimplemented!")
            elif op == Op.XOR:
                registers[rd] = registers[rs1] ^ registers[rs2]
            elif op == Op.SRL_SRA:
                shamt = get_bits(registers[rs2], 4, 0)
                if funct7 == 0b0000000:
                    # SRL instruction
                    registers[rd] = registers[rs1] >> shamt
                elif funct7 == 0b0100000:
                    # SRA instruction
                    registers[rd] = sign_extend(
                        registers[rs1] >> shamt, 32 - shamt, 32
                    )
            elif op == Op.OR:
                registers[rd] = registers[rs1] | registers[rs2]
            elif op == Op.AND:
                registers[rd] = registers[rs1] & registers[rs2]

        # UNIMPLEMENTED
        elif opcode == Opcode.MISC_MEM:
            # FENCE instruction
            print(f'')

        # Partially UNIMPLEMENTED
        elif opcode == Opcode.SYSTEM:
            # ECALL, EBREAK instructions
            # I type
            funct12 = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)

            try:
                system_op = SystemOp(funct12)
                print(
                    f'{system_op.name.rjust(6)} {funct12=} {rs1=} {funct3=} {rd=}'
                )

                if system_op == SystemOp.ECALL:
                    print('Syscall requested, are we good?')
                elif system_op == SystemOp.EBREAK:
                    raise Exception("Unimplemented!")
            except ValueError:
                # One of the CSR instructions, ignore for now
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
        for x in glob.glob('./riscv-tests/isa/rv32ui-p-addi')
        if len(x.split('.')) == 2
    ]
    for path in paths:
        print(f'Executing file: {path}')
        with open(path, 'rb') as file:
            execute_elf(file)
            break
