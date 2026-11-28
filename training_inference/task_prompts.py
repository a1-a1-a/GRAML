from __future__ import annotations

from typing import Any, Dict


DETECTION = (
    "You are a security code vulnerability analyzer. Your task is to carefully analyze the provided code snippet. "
    "Note that the provided code snippet might not be complete, but it has all the important context. "
    "Your output must be EXACTLY ONE WORD: If you detect any potential security vulnerability in the specified code segment, return: Yes. "
    "If the code segment appears to be secure and free from obvious vulnerabilities, return: No. "
    "Consider common vulnerability types such as buffer overflows, improper input validation, integer overflow, memory corruption potential, double free, and use after free. "
    'Your response must be either "Yes" or "No" with no additional explanation.'
)

DESCRIPTION = (
    "You are a security code vulnerability analyzer. Your task is to analyze the following code and determine whether it contains vulnerabilities. "
    "Write the final vulnerability explanation as one coherent paragraph in professional security-analysis style that clearly states what the vulnerability is, "
    "why it occurs as the root cause, how it can be triggered, how the internal logic or data or control flow leads to the flaw, and what impact an attacker may achieve. "
    "The output must be a single paragraph of 3 to 5 sentences, must not contain any parentheses, and may quote small relevant portions of the source code "
    "without including line numbers or bracket references. The writing must be fluent, concise, and technically professional. "
    'If you determine that there is no vulnerability, output only "There is no vulnerability."'
)

LOCATION = (
    "You are a security code vulnerability analyzer. Your task is to carefully analyze the following C code and identify the exact locations of potential vulnerabilities. "
    "If any vulnerabilities exist, output the zero-based vulnerable line indices separated by commas. "
    "Consider the vulnerable statements, operations, and code regions that directly contribute to the security issue. "
    'Output format example: 1,4,7. If no vulnerabilities are found, only output "N/A". Do not provide any additional explanation.'
)

ASSESSMENT = (
    "You are a security code vulnerability analyzer. Your task is to analyze the following C code snippet and provide the CVSS v3.0 severity rating "
    "based on the identified vulnerability. Use the following mapping: High or Critical = 2, Medium = 1, Low = 0. "
    "Consider the potential impact and exploitability of the vulnerability when assigning the severity level. "
    'Output only one digit, either 2 or 1 or 0, and do not output any other text. If you determine that there is no vulnerability, only output "N/A".'
)

TASK_INSTRUCTIONS = {
    "detection": DETECTION,
    "description": DESCRIPTION,
    "location": LOCATION,
    "localization": LOCATION,
    "assessment": ASSESSMENT,
}

SYSTEM_PROMPT = "You are a security assistant for vulnerability detection, assessment, location and description."

DIRECT_CPG_MARKER = "output its raw Joern CPG edge records"


def build_prompt_mistral(system_prompt: str, instruction: str, user_input: str) -> str:
    user = instruction.strip()
    if user_input:
        user = user + "\n## Input:\n" + str(user_input).strip()
    return f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>\n{user} [/INST]"


def build_prompt_deepseek(system_prompt: str, instruction: str, user_input: str) -> str:
    user = instruction.strip()
    if user_input:
        user = user + "\n" + str(user_input).strip()
    return f"{system_prompt}\n### Instruction:\n{user}\n### Response:\n"


def build_prompt_alpaca(system_prompt: str, instruction: str, user_input: str) -> str:
    user = instruction.strip()
    input_str = ""
    if user_input:
        input_str = f"\n\n### Input:\n{str(user_input).strip()}"
    return f"{system_prompt}\n\n### Instruction:\n{user}{input_str}\n\n### Response:\n"


def build_prompt_qwen(system_prompt: str, instruction: str, user_input: str) -> str:
    user = instruction.strip()
    if user_input:
        user = user + "\n" + str(user_input).strip()
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


PROMPT_BUILDERS = {
    "mistral": build_prompt_mistral,
    "deepseek": build_prompt_deepseek,
    "alpaca": build_prompt_alpaca,
    "qwen": build_prompt_qwen,
}


def infer_task(sample: Dict[str, Any]) -> str:
    task = sample.get("Task", sample.get("task", sample.get("category", sample.get("type", ""))))
    task = str(task).strip().lower()
    if task in TASK_INSTRUCTIONS:
        return "location" if task == "localization" else task
    instruction = str(sample.get("instruction", sample.get("Instruction", ""))).lower()
    if "zero-based vulnerable line" in instruction or "exact locations" in instruction:
        return "location"
    if "cvss v3.0" in instruction or "severity rating" in instruction:
        return "assessment"
    if "vulnerability explanation" in instruction or "there is no vulnerability" in instruction:
        return "description"
    return "detection"


def canonical_instruction(task: str, fallback: str = "") -> str:
    key = str(task).strip().lower()
    if key == "localization":
        key = "location"
    return TASK_INSTRUCTIONS.get(key, fallback)


def resolve_instruction(sample: Dict[str, Any], task: str | None = None) -> str:
    instruction = str(sample.get("instruction", sample.get("Instruction", "")) or "")
    if (
        sample.get("_target_representation") == "joern_cpg_edge_records"
        or DIRECT_CPG_MARKER in instruction
    ):
        return instruction
    return canonical_instruction(task if task is not None else infer_task(sample), instruction)
