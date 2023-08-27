from typing import List
import subprocess
import pytest

command = ["python", "risc_v_python/main.py", "-q"]

tests_path = './riscv-tests/isa'

instructions = [
    "blt",
    "slli",
    "beq",
    "lbu",
    "lh",
    "sra",
    "lb",
    "xori",
    "slt",
    "sh",
    "sltiu",
    "add",
    "and",
    "slti",
    "srl",
    "lw",
    "bltu",
    "sub",
    "simple",
    "sll",
    "andi",
    "srai",
    "jal",
    "xor",
    "sltu",
    "ori",
    "bge",
    "addi",
    "lui",
    "auipc",
    "lhu",
    "srli",
    "jalr",
    "bgeu",
    # "fence_i",
    "bne",
    "sb",
    "sw",
    "or", 
]

def get_test_command(file: str) -> List[str]:
    local_command = command.copy()
    local_command.append(file)
    return local_command

@pytest.mark.parametrize('ins', instructions)
def test_instruction(ins: str):
    p = subprocess.run(get_test_command(f"{tests_path}/rv32ui-p-{ins}"))
    assert p.returncode == 0

