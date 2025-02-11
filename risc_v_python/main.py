import sys
from enum import Enum, auto
import struct
import numpy as np
from elftools.elf.elffile import ELFFile
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.containers import Container
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from risc_v_python.constants import (
    BranchOp,
    LoadOp,
    Op,
    Opcode,
    OpImm,
    StoreOp,
    SystemOp,
    CsrOp,
)


class FormatType(Enum):
    NUMBER = auto()
    ADDR = auto()
    OP = auto()


class InsArgs:
    op_imm: OpImm
    load_op: LoadOp
    system_op: SystemOp
    op: Op
    csr_op: CsrOp
    imm: int
    rs1: int
    rs2: int
    rd: int
    funct3: int
    funct7: int
    funct12: int
    csr: int


class Instruction:
    addr: int
    ins: int
    opcode: Opcode
    args: InsArgs
    unimp: bool = False

    def format_ins_arg(self, arg_name, type=FormatType.NUMBER):
        if hasattr(self.args, arg_name):
            arg = getattr(self.args, arg_name)
            if type == FormatType.OP:
                formatted = f"{arg.name.ljust(9)}"
            elif type == FormatType.ADDR:
                formatted = f"[yellow]{arg_name}[/][bright_black]=[/]{arg:08x}"
            else:
                formatted = f"[yellow]{arg_name}[/][bright_black]=[/]{arg}"
        else:
            formatted = ""
        return formatted

    def __str__(self):
        op_desc = "".join(
            [
                self.format_ins_arg("op_imm", FormatType.OP),
                self.format_ins_arg("load_op", FormatType.OP),
                self.format_ins_arg("op", FormatType.OP),
                self.format_ins_arg("system_op", FormatType.OP),
                self.format_ins_arg("csr_op", FormatType.OP),
                "[red]UNIMP    [/]" if self.unimp else "",
            ]
        )
        if op_desc == "":
            op_desc = "".ljust(9)

        info = [
            f"[not bold green]{self.addr:08x}[/]",
            f"[not bold cyan]{self.ins:08x}[/]",
            self.opcode.name.ljust(8),
            op_desc,
            self.format_ins_arg("funct3"),
            self.format_ins_arg("funct7"),
            self.format_ins_arg("funct12"),
            self.format_ins_arg("csr"),
            self.format_ins_arg("imm"),
            self.format_ins_arg("rs1"),
            self.format_ins_arg("rs2"),
            self.format_ins_arg("rd"),
        ]
        filtered = filter(lambda x: x != "", info)
        return " ".join(filtered)


def get_mask(start, end):
    return ((1 << (start + 1)) - 1) - ((1 << end) - 1)


def get_bits(value, start, end):
    return (value & get_mask(start, end)) >> end


def sign_extend(value, cur_size, new_size):
    if value & get_mask(cur_size - 1, cur_size - 1):
        # If negative, sign extend
        value = get_mask(new_size - 1, cur_size) | value
    return value


def format_register(reg_name, value):
    return f"[bold red]{reg_name}[/][bright_black]=[/]{value:08x} "


def format_arg(arg, type=FormatType.NUMBER):
    name, value = f"{arg=}".split("=")
    if type == FormatType.ADDR:
        value_formatted = f"{int(value):08x}"
    else:
        value_formatted = value

    return f"[yellow]{name}[/]=[blue]{value_formatted}[/]"


class Registers:
    """
    Provides access to registers
    """

    def __init__(self):
        self.__registers = np.zeros(32, dtype=np.uint32)

    def __getitem__(self, index):
        return 0 if index == 0 else self.__registers[index]

    def __setitem__(self, index, value):
        if index != 0:
            self.__registers[index] = value

    def dump_regs(self, regs_per_line=4):
        result = ""
        for i in range(32):
            result += format_register(f"x{i:02}", self.__registers[i])
            if (i % regs_per_line == regs_per_line - 1) and i != 31:
                result += "\n"
        return result

class CSR:
    """
    Provides access to control and status registers (CSR).
    """

    def __init__(self):
        self.__csr = np.zeros(4096, dtype=np.uint32)

    def __getitem__(self, index):
        return self.__csr[index]

    def __setitem__(self, index, value):
        self.__csr[index] = value

class Emulator:
    """
    Virtual RISC-V machine
    """

    pc = np.uint32(0)
    start_addr = 0
    offset = 0
    size = 0
    registers = Registers()
    csr = CSR()
    # Allocate 4 KB heap
    # Memory is unimplemented yet
    memory = np.zeros(4 * 1024, dtype=np.uint8)
    finished = False

    def load_elf(self, filename):
        """
        Load an ELF program
        """

        with open(filename, "rb") as file:
            elf = ELFFile(file)
            text_init = elf.get_section_by_name(".text.init")
            if text_init is None:
                raise Exception("ELF is empty")
            self.start_addr = np.uint32(text_init.header["sh_addr"])
            self.data = text_init.data()
            self.size = text_init.header["sh_size"]

        self.pc = self.start_addr
        self.offset = 0
        self.finished = False

    def next_ins(self):
        """
        Execute next instruction
        """

        if self.offset + 4 <= self.size:
            binary_ins = self.data[self.offset : self.offset + 4]
            ins = struct.unpack("I", binary_ins)[0]
            try:
                instruction = self.execute_instruction(ins)
                # Normal flow, go to the next instruction
                self.pc += 4
                self.offset = self.pc - self.start_addr
                return str(instruction)
            except Exception as e:
                self.finished = True
                return f"[red]{str(e)}[/red]"

    def execute_instruction(self, ins) -> Instruction:
        """
        Execute a single instruction
        """
        opcode = Opcode(get_bits(ins, 6, 0))

        instruction = Instruction()
        instruction.addr = self.pc.item()
        instruction.ins = ins
        instruction.opcode = opcode
        instruction.args = InsArgs()

        if opcode == Opcode.LUI:
            # U type
            imm = get_bits(ins, 31, 12) << 12
            rd = get_bits(ins, 11, 7)

            instruction.args.imm = imm
            instruction.args.rd = rd

            self.registers[rd] = imm

        elif opcode == Opcode.AUIPC:
            # U type
            imm = get_bits(ins, 31, 12) << 12
            rd = get_bits(ins, 11, 7)

            instruction.args.imm = imm
            instruction.args.rd = rd

            self.registers[rd] = self.pc + imm

        elif opcode == Opcode.JAL:
            # J type
            imm = (
                (get_bits(ins, 31, 31) << 20)
                | (get_bits(ins, 30, 21) << 1)
                | (get_bits(ins, 20, 20) << 11)
                | (get_bits(ins, 19, 12) << 12)
            )
            rd = get_bits(ins, 11, 7)

            instruction.args.imm = imm
            instruction.args.rd = rd

            # Sign extending the immediate
            imm = sign_extend(imm, 32, 32)
            self.registers[rd] = self.pc + 4
            self.pc = self.pc + imm - 4

        elif opcode == Opcode.JALR:
            # I type
            imm = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)

            instruction.args.imm = imm
            instruction.args.rs1 = rs1
            instruction.args.funct3 = funct3
            instruction.args.rd = rd

            # TODO: Let's hope it works, need to test
            # Sign extending the immediate
            imm = sign_extend(imm, 12, 32)

            branch_to = ((self.registers[rs1] + imm) & get_mask(31, 1)) - 4
            self.registers[rd] = self.pc + 4
            self.pc = branch_to

        elif opcode == Opcode.BRANCH:
            # BEQ, BNE, BLT, BGE, BLTU, BGEU instructions
            # B type
            imm = (
                (get_bits(ins, 31, 31) << 20)
                | (get_bits(ins, 30, 25) << 5)
                | (get_bits(ins, 11, 8) << 1)
                | (get_bits(ins, 7, 7) << 11)
            )
            rs2 = get_bits(ins, 24, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)

            # TODO: Let's hope it works, need to test
            # Sign extending the immediate
            imm = sign_extend(imm, 12, 32)

            branch_to = (self.pc + imm - 4) & get_mask(31, 0)

            branch_op = BranchOp(funct3)

            instruction.args.imm = imm
            instruction.args.rs1 = rs1
            instruction.args.rs2 = rs2
            instruction.args.rd = rd

            if branch_op == BranchOp.BEQ:
                if self.registers[rs1] == self.registers[rs2]:
                    self.pc = branch_to
            elif branch_op == BranchOp.BNE:
                if self.registers[rs1] != self.registers[rs2]:
                    self.pc = branch_to
            elif branch_op == BranchOp.BLT:
                if np.int32(self.registers[rs1]) < np.int32(self.registers[rs2]):
                    self.pc = branch_to
            elif branch_op == BranchOp.BGE:
                if np.int32(self.registers[rs1]) < np.int32(self.registers[rs2]):
                    self.pc = branch_to
            elif branch_op == BranchOp.BLTU:
                if self.registers[rs1] < self.registers[rs2]:
                    self.pc = branch_to
            elif branch_op == BranchOp.BGEU:
                if self.registers[rs1] > self.registers[rs2]:
                    self.pc = branch_to
            else:
                # No instruction with such branch_op
                raise Exception(f"Wrong {branch_op=:03b} for branch instructions!")

        # UNIMPLEMENTED
        elif opcode == Opcode.LOAD:
            # LB, LH, LW, LBU, LHU instructions
            # I type
            imm = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)
            load_op = LoadOp(funct3)

            instruction.args.load_op = load_op
            instruction.args.imm = imm
            instruction.args.rs1 = rs1
            instruction.args.rd = rd

        # UNIMPLEMENTED
        elif opcode == Opcode.STORE:
            # SB, SH, SW instructions
            # S type
            instruction.unimp = True

        # Partially UNIMPLEMENTED
        elif opcode == Opcode.OP_IMM:
            # Arithmetic instrutions with immediate (9 pcs)
            imm = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)

            op_imm = OpImm(funct3)

            instruction.args.op_imm = op_imm
            instruction.args.imm = imm
            instruction.args.rs1 = rs1
            instruction.args.rd = rd

            if op_imm == OpImm.ADDI:
                imm = sign_extend(imm, 12, 32)
                self.registers[rd] = self.registers[rs1] + imm
            elif op_imm == OpImm.SLTI:
                imm = sign_extend(imm, 12, 32)
                self.registers[rd] = int(self.registers[rs1] < imm)
            elif op_imm == OpImm.SLTIU:
                imm = sign_extend(imm, 12, 32)
                raise Exception("Unimplemented!")
            elif op_imm == OpImm.XORI:
                imm = sign_extend(imm, 12, 32)
                self.registers[rd] = self.registers[rs1] ^ imm
            elif op_imm == OpImm.ORI:
                imm = sign_extend(imm, 12, 32)
                self.registers[rd] = self.registers[rs1] | imm
            elif op_imm == OpImm.ANDI:
                imm = sign_extend(imm, 12, 32)
                self.registers[rd] = self.registers[rs1] & imm
            elif op_imm == OpImm.SLLI:
                shamt = get_bits(imm, 4, 0)
                self.registers[rd] = self.registers[rs1] << shamt
            elif op_imm == OpImm.SRLI_SRAI:
                shamt = get_bits(imm, 4, 0)
                funct7 = get_bits(imm, 11, 5)

                instruction.args.funct7 = funct7

                if funct7 == 0b0000000:
                    # SRLI instruction
                    self.registers[rd] = self.registers[rs1] >> shamt
                elif funct7 == 0b0100000:
                    # SRAI instruction
                    self.registers[rd] = sign_extend(
                        self.registers[rs1] >> shamt, 32 - shamt, 32
                    )
                else:
                    raise Exception("Unknown funct7 for SRLI_SRAI!")

        # Partially UNIMPLEMENTED
        elif opcode == Opcode.OP:
            # Arithmetic instrutions (10 pcs)
            funct7 = get_bits(ins, 31, 25)
            rs2 = get_bits(ins, 24, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)

            op = Op(funct3)

            instruction.args.op = op
            instruction.args.funct7 = funct7
            instruction.args.rs1 = rs1
            instruction.args.rs2 = rs2
            instruction.args.rd = rd

            if op == Op.ADD_SUB:
                if funct7 == 0b0000000:
                    # ADD instruction
                    self.registers[rd] = self.registers[rs1] + self.registers[rs2]
                elif funct7 == 0b0100000:
                    # SUB instruction
                    self.registers[rd] = self.registers[rs1] - self.registers[rs2]
            elif op == Op.SLL:
                shamt = get_bits(self.registers[rs2], 4, 0)
                self.registers[rd] = self.registers[rs1] << shamt
            elif op == Op.SLT:
                self.registers[rd] = int(self.registers[rs1] < self.registers[rs2])
            elif op == Op.SLTU:
                raise Exception("Unimplemented!")
            elif op == Op.XOR:
                self.registers[rd] = self.registers[rs1] ^ self.registers[rs2]
            elif op == Op.SRL_SRA:
                shamt = get_bits(self.registers[rs2], 4, 0)
                if funct7 == 0b0000000:
                    # SRL instruction
                    self.registers[rd] = self.registers[rs1] >> shamt
                elif funct7 == 0b0100000:
                    # SRA instruction
                    self.registers[rd] = sign_extend(
                        self.registers[rs1] >> shamt, 32 - shamt, 32
                    )
            elif op == Op.OR:
                self.registers[rd] = self.registers[rs1] | self.registers[rs2]
            elif op == Op.AND:
                self.registers[rd] = self.registers[rs1] & self.registers[rs2]

        # UNIMPLEMENTED
        elif opcode == Opcode.MISC_MEM:
            # FENCE instruction
            instruction.unimp = True

        # Partially UNIMPLEMENTED
        elif opcode == Opcode.SYSTEM:
            # ECALL, EBREAK instructions
            # I type
            imm = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)

            instruction.args.rs1 = rs1
            instruction.args.funct3 = funct3
            instruction.args.rd = rd

            if funct3 == 0b000 and rs1 == 0b00000 and rd == 0b00000:
                try:
                    system_op = SystemOp(imm)

                    instruction.args.funct12 = imm
                    instruction.args.system_op = system_op

                    syscall = self.registers[17]

                    if system_op == SystemOp.ECALL:
                        # registers 10 - 17 are used for syscalls
                        if syscall == 93:
                            # exit() syscall
                            raise Exception(f"Program finished, return value {self.registers[10]}.")
                        if self.registers[10] == 1:
                            # Test passed!
                            raise Exception("Test passed!")
                        else:
                            raise Exception("Test failed!")
                    elif system_op == SystemOp.EBREAK:
                        raise Exception("Debug breakpoint!")
                except ValueError:
                    # MRET instruction calls this.
                    # noop seems to work.
                    instruction.unimp = True
            else:
                # One of the CSR instructions, ignore for now
                csr_op = CsrOp(funct3)

                instruction.args.csr = imm
                instruction.args.csr_op = csr_op

                if csr_op == CsrOp.CSRRW:
                    if instruction.args.rd != 0:
                        self.registers[instruction.args.rd] = self.csr[instruction.args.csr]
                    self.csr[instruction.args.csr] = self.registers[instruction.args.rs1]
                elif csr_op == CsrOp.CSRRS:
                    self.registers[instruction.args.rd] = self.csr[instruction.args.csr]
                    if instruction.args.rs1 != 0:
                        self.csr[instruction.args.csr] |= self.registers[instruction.args.rs1]
                elif csr_op == CsrOp.CSRRC:
                    self.registers[instruction.args.rd] = self.csr[instruction.args.csr]
                    if instruction.args.rs1 != 0:
                        self.csr[instruction.args.csr] &= ~self.registers[instruction.args.rs1]
                elif csr_op == CsrOp.CSRRWI:
                    if instruction.args.rd != 0:
                        self.registers[instruction.args.rd] = self.csr[instruction.args.csr]
                    self.csr[instruction.args.csr] = instruction.args.rs1
                elif csr_op == CsrOp.CSRRSI:
                    self.registers[instruction.args.rd] = self.csr[instruction.args.csr]
                    if instruction.args.rs1 != 0:
                        self.csr[instruction.args.csr] |= instruction.args.rs1
                elif csr_op == CsrOp.CSRRCI:
                    self.registers[instruction.args.rd] = self.csr[instruction.args.csr]
                    if instruction.args.rs1 != 0:
                        self.csr[instruction.args.csr] &= ~instruction.args.rs1

        else:
            raise Exception("UNKNOWN")

        return instruction


class EmulatorApp(App):

    body = Container(Static(Panel("")), id="body")
    registers = Static(Panel(""), id="registers")
    ins_output = ""

    TITLE = "RISC-V Emulator"
    CSS_PATH = "main.tcss"
    BINDINGS = [
        ("n", "next_ins", "Next"),
        ("j", "scroll_down", "Scroll down"),
        ("k", "scroll_up", "Scroll up"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """
        Create and dock the widgets
        """
        yield self.registers
        yield self.body
        yield Header()
        yield Footer()

    def on_mount(self) -> None:

        def prepare_ui():
            self.update_ui("File loaded, waiting to execute next instruction...\n")
            self.ins_output = ""

        self.call_later(prepare_ui)

    def action_next_ins(self) -> None:
        if not emulator.finished:
            result = emulator.next_ins()
            self.update_ui(result)

    def update_ui(self, result) -> None:
        self.ins_output += f"{result}\n"
        self.body.children[0].update(self.ins_output)

        self.registers.update(
            Panel(
                Text.from_markup(
                    f"{format_register('PC', emulator.pc)}\n\n{emulator.registers.dump_regs(regs_per_line=2)}"
                ),
                title="Registers",
            )
        )
        self.body.scroll_down()

    def action_scroll_up(self) -> None:
        self.body.scroll_relative(
            y=-1,
            animate=False,
        )

    def action_scroll_down(self) -> None:
        self.body.scroll_relative(y=1, animate=False)


if __name__ == "__main__":
    # Ignore numpy overflow warnings.
    np.seterr(over="ignore")

    if (len(sys.argv) < 2):
        print("Usage: [program_name] [riscv_program].")
        sys.exit(1)

    emulator = Emulator()
    args = [x for x in sys.argv[1:] if x != "-q"]
    emulator.load_elf(args[0])

    if (len(args) < len(sys.argv[1:])):
        # Run in quiet mode
        ins_text = ""
        console = Console()
        while not emulator.finished:
            ins_text = emulator.next_ins()
            console.print(ins_text)
        if ins_text == "[red]Program finished, return value 0.[/red]":
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        app = EmulatorApp()
        app.run()
