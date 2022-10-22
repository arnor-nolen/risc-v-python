import glob
import struct
import numpy as np
from elftools.elf.elffile import ELFFile
from textual.app import App
from textual import events
from textual.reactive import Reactive
from textual.widgets import Header, Footer, Placeholder, ScrollView, Static
from textual.widget import Widget
from textual.views import GridView, WindowView
from textual.reactive import watch
from textual.scrollbar import ScrollTo
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from risc_v_python.constants import (
    BranchOp,
    LoadOp,
    Op,
    Opcode,
    OpImm,
    StoreOp,
    SystemOp,
)


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
    return f'[bold red]{reg_name}[/bold red][bright_black]=[/bright_black]{value:08x} '


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
        result = ''
        for i in range(32):
            result += format_register(f'x{i:02}', self.__registers[i])
            if i % regs_per_line == regs_per_line - 1:
                result += '\n'
        return result


class Emulator:
    """
    Virtual RISC-V machine
    """

    pc = 0
    start_addr = 0
    offset = 0
    size = 0
    registers = Registers()
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
            text_init = elf.get_section_by_name('.text.init')
            self.start_addr = np.uint32(text_init.header['sh_addr'])
            self.data = text_init.data()
            self.size = text_init.header['sh_size']

        self.pc = self.start_addr
        self.offset = 0
        self.finished = False

    def next_ins(self):
        """
        Execute next instruction
        """

        if self.offset + 4 <= self.size:
            binary_ins = self.data[self.offset : self.offset + 4]
            ins = struct.unpack('I', binary_ins)[0]
            try:
                output = self.execute_instruction(ins)
                # Normal flow, go to the next instruction
                self.pc += 4
                self.offset = self.pc - self.start_addr
                return output
            except Exception as e:
                self.finished = True
                return f'[red]{str(e)}[/red]'

    def execute_instruction(self, ins) -> str:
        """
        Execute a single instruction
        """
        opcode = Opcode(get_bits(ins, 6, 0))

        output = f'[not bold green]{self.pc:08x}[/not bold green] [not bold cyan]{ins:08x}[/not bold cyan] {opcode.name.ljust(6)} '

        if opcode == Opcode.LUI:
            # U type
            imm = get_bits(ins, 31, 12) << 12
            rd = get_bits(ins, 11, 7)
            output += f'{"".ljust(9)} {imm=:08x} {rd=}\n'

            self.registers[rd] = imm

        elif opcode == Opcode.AUIPC:
            # U type
            imm = get_bits(ins, 31, 12) << 12
            rd = get_bits(ins, 11, 7)
            output += f'{"".ljust(9)} {imm=:08x} {rd=}\n'

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
            output += f'{"".ljust(9)} {imm=:08x} {rd=}\n'

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
            output += f'{"".ljust(9)} {imm=:08x} {rs1=} {funct3=} {rd=}\n'

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
            output += (
                f'{branch_op.name.ljust(9)} {imm=:08x} {rs1=} {rs2=} {rd=}\n'
            )

            if branch_op == BranchOp.BEQ:
                if self.registers[rs1] == self.registers[rs2]:
                    self.pc = branch_to
            elif branch_op == BranchOp.BNE:
                if self.registers[rs1] != self.registers[rs2]:
                    self.pc = branch_to
            elif branch_op == BranchOp.BLT:
                if self.registers[rs1] & get_mask(31, 0) < self.registers[
                    rs2
                ] & get_mask(31, 0):
                    self.pc = branch_to
            elif branch_op == BranchOp.BGE:
                if self.registers[rs1] & get_mask(31, 0) < self.registers[
                    rs2
                ] & get_mask(31, 0):
                    self.pc = branch_to
            elif branch_op == BranchOp.BLTU:
                if self.registers[rs1] < self.registers[rs2]:
                    self.pc = branch_to
            elif branch_op == BranchOp.BGEU:
                if self.registers[rs1] > self.registers[rs2]:
                    self.pc = branch_to
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

            output += f'{load_op.name.ljust(9)} {imm=:08x} {rs1=} {rd=}\n'

        # UNIMPLEMENTED
        elif opcode == Opcode.STORE:
            # SB, SH, SW instructions
            # S type
            output += f'\n'

        # Partially UNIMPLEMENTED
        elif opcode == Opcode.OP_IMM:
            # Arithmetic instrutions with immediate (9 pcs)
            imm = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)

            op_imm = OpImm(funct3)

            output += f'{op_imm.name.ljust(9)} {imm=:08x} {rs1=} {rd=}\n'
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

            print(f'{op.name.ljust(9)} {funct7=} {rs1=} {rs2=} {rd=}')

            if op == Op.ADD_SUB:
                if funct7 == 0b0000000:
                    # ADD instruction
                    self.registers[rd] = (
                        self.registers[rs1] + self.registers[rs2]
                    )
                elif funct7 == 0b0100000:
                    # SUB instruction
                    self.registers[rd] = (
                        self.registers[rs1] - self.registers[rs2]
                    )
            elif op == Op.SLL:
                shamt = get_bits(self.registers[rs2], 4, 0)
                self.registers[rd] = self.registers[rs1] << shamt
            elif op == Op.SLT:
                self.registers[rd] = int(
                    self.registers[rs1] < self.registers[rs2]
                )
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
            output += f'\n'

        # Partially UNIMPLEMENTED
        elif opcode == Opcode.SYSTEM:
            # ECALL, EBREAK instructions
            # I type
            funct12 = get_bits(ins, 31, 20)
            rs1 = get_bits(ins, 19, 15)
            funct3 = get_bits(ins, 14, 12)
            rd = get_bits(ins, 11, 7)

            if funct3 == 0b000 and rs1 == 0b00000 and rd == 0b00000:
                try:
                    system_op = SystemOp(funct12)
                    output += f'{system_op.name.ljust(9)} {funct12=} {rs1=} {funct3=} {rd=}\n'

                    if system_op == SystemOp.ECALL:
                        # registers 10 - 17 are used for syscalls
                        if self.registers[10] == 1:
                            # Test passed!
                            raise Exception("Test passed!")
                        else:
                            raise Exception("Test failed!")
                    elif system_op == SystemOp.EBREAK:
                        raise Exception("Unimplemented!")
                except ValueError:
                    output += f'{"UNKNOWN".ljust(9)}\n'
            else:
                # One of the CSR instructions, ignore for now
                output += f'{"CSR".ljust(9)} [red]UNIMPLEMENTED[/red]\n'

        else:
            output += f'UNKNOWN\n'

        return output


class EmulatorApp(App):

    registers = Static(Panel(''))
    ins_output = ''

    async def on_load(self, event: events.Load) -> None:
        """
        Bind keys with the app loads (but before entering application mode)
        """
        await self.bind("n", "next_ins", "Next")
        await self.bind("j", "scroll_down", "Scroll down")
        await self.bind("k", "scroll_up", "Scroll up")
        await self.bind("q", "quit", "Quit")

        self.body = ScrollView(Panel(''))

    async def on_mount(self, event: events.Mount) -> None:
        """
        Create and dock the widgets
        """

        await self.view.dock(Header(clock=False), edge="top")
        await self.view.dock(Footer(), edge="bottom")
        await self.view.dock(self.registers, edge="right", size=29)
        await self.view.dock(self.body, edge="left")

        filename = "./riscv-tests/isa/rv32ui-p-add"

        async def prepare_ui(filename):
            emulator.load_elf(filename)
            await self.update_ui(
                "File loaded, waiting to execute next instruction...\n"
            )
            self.ins_output = ''

        await self.call_later(prepare_ui, filename)

    async def action_next_ins(self):
        if not emulator.finished:
            result = emulator.next_ins()
            await self.update_ui(result)

    async def update_ui(self, result):
        self.ins_output += result
        await self.body.update(self.ins_output, home=False)
        await self.registers.update(
            Panel(
                f"{format_register('PC', emulator.pc)}\n\n{emulator.registers.dump_regs(regs_per_line=2)}",
                title="Registers",
            )
        )
        await self.body.vscroll.action_scroll_down()

    async def action_scroll_up(self):
        await self.body.vscroll.emit(
            ScrollTo(self.body.vscroll, y=self.body.vscroll.position - 1)
        )

    async def action_scroll_down(self):
        await self.body.vscroll.emit(
            ScrollTo(self.body.vscroll, y=self.body.vscroll.position + 1)
        )


if __name__ == '__main__':
    emulator = Emulator()
    EmulatorApp.run(title="RISC-V Emulator")
    # paths = [
    #     x
    #     for x in glob.glob('./riscv-tests/isa/rv32ui-p-*')
    #     if not x.endswith('.dump')
    # ]
    # for path in paths:
    #     print(f'Executing file: {path}')
    #     with open(path, 'rb') as file:
    #         execute_elf(file)
