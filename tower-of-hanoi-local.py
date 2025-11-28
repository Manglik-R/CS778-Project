#!/usr/bin/env python3
"""
Tower-of-Hanoi experiment on a LOCAL DeepSeek-R1 model.

- System & user prompts are consistent with:
  * 3 pegs, 0-indexed
  * disks numbered 1 (smallest) .. N (largest)
  * final answer format: moves = [[disk id, from peg, to peg], ...]
- Uses local Hugging Face model (no external API).
- max_new_tokens per trial = min(2000 * N, 10000).
- Uses sampling (temperature/top_p) + random seed tag so each trial is different.
- Validates the moves by simulating Hanoi.
- Stores only a JSON summary (no CSV).
"""

import os
import json
import random
import re
import ast

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------- CONFIG --------
MODEL_NAME_OR_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# Start smaller for debugging; you can later use range(1, 11)
DISC_SIZES = [8, 12, 16, 20]
TRIALS_PER_SIZE = 1

OUT_SUMMARY = "./game1/summary.json"

# -------- SYSTEM PROMPT (your spec) --------
SYSTEM_PROMPT = """
You are a helpful assistant. Solve this Tower of Hanoi puzzle for me.

There are three pegs and n disks of different sizes stacked on the first peg.
The disks are numbered from 1 (smallest) to n (largest).

Disk moves in this puzzle must follow:
1. Only one disk can be moved at a time.
2. Each move consists of taking the upper disk from one stack and placing it on top of another stack.
3. A larger disk may not be placed on top of a smaller disk.

The goal is to move the entire stack to the third peg.

The positions are 0-indexed (the leftmost peg is 0, then 1, then 2).

Example (n = 3):
- Initial state: [[3, 2, 1], [], []]
- One valid solution is:
    moves = [[1, 0, 2],
             [2, 0, 1],
             [1, 2, 1],
             [3, 0, 2],
             [1, 1, 0],
             [2, 1, 2],
             [1, 0, 2]]

This means: move disk 1 from peg 0 to peg 2, then disk 2 from peg 0 to peg 1, and so on.

Requirements:
- When exploring potential solutions in your thinking process, you may reason step-by-step.
- The final answer MUST include the complete list of moves in the format:
  moves = [[disk id, from peg, to peg], ...]
- Use only integers 1..n for disk ids, and 0, 1, 2 for peg indices.
"""

# -------- USER PROMPT TEMPLATE --------
def make_user_prompt(n: int) -> str:
    return f"""
I have a Tower of Hanoi puzzle with {n} disks of different sizes.

Initial configuration:
- Peg 0: {n} (bottom), ..., 2, 1 (top)
- Peg 1: (empty)
- Peg 2: (empty)

Goal configuration:
- Peg 0: (empty)
- Peg 1: (empty)
- Peg 2: {n} (bottom), ..., 2, 1 (top)

Rules:
- Only one disk can be moved at a time.
- Only the top disk from any stack can be moved.
- A larger disk may not be placed on top of a smaller disk.

Find the sequence of moves to transform the initial configuration into the goal configuration.

Return your final answer in the EXACT format:
moves = [[disk id, from peg, to peg], ...]
"""


# -------- LOAD MODEL LOCALLY --------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
device = model.device
print(f"Model loaded on device: {device}")


# -------- MODEL CALL --------
def call_local_model(system_prompt: str, user_prompt: str, max_new_tokens: int):
    """
    Calls the local DeepSeek-R1 model and returns (text, usage_dict).
    Uses sampling to get diverse trials.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Build prompt string via chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback if the tokenizer has no chat template
        prompt_str = (
            "System:\n" + system_prompt.strip() + "\n\n"
            "User:\n" + user_prompt.strip() + "\n\n"
            "Assistant:\n"
        )

    inputs = tokenizer(prompt_str, return_tensors="pt", padding=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    input_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,        # enable sampling so trials differ
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    total_len = output_ids.shape[-1]
    output_len = max(total_len - input_len, 0)

    gen_ids = output_ids[0, input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    usage = {
        "input_tokens": int(input_len),
        "output_tokens": int(output_len),
        "total_tokens": int(total_len),
    }
    return text, usage


# -------- PARSE moves = [[...], ...] --------
def extract_moves(text: str):
    """
    Extracts the Python-like list from an expression of the form:
       moves = [[disk, from, to], ...]
    Returns (moves_list, parse_error).
    """
    moves = None
    parse_error = None

    # Regex to find 'moves = [[...]]' capturing the list part
    m = re.search(r"moves\s*=\s*(\[\s*\[.*\]\s*\])", text, re.DOTALL)
    if not m:
        return None, "No 'moves = [...]' pattern found"

    list_str = m.group(1)
    try:
        # Safely evaluate the list literal
        moves = ast.literal_eval(list_str)
    except Exception as exc:
        parse_error = f"literal_eval error: {exc}"
        return None, parse_error

    # Basic sanity checks
    if not isinstance(moves, list):
        return None, "Parsed 'moves' is not a list"

    for idx, mv in enumerate(moves):
        if not isinstance(mv, (list, tuple)) or len(mv) != 3:
            return None, f"Move {idx+1} is not a length-3 list"

    return moves, None


# -------- VALIDATE HANOI SEQUENCE --------
def validate_hanoi(n: int, moves):
    """
    Simulates the moves on 3 pegs:
      - Pegs 0, 1, 2
      - Disks 1..n where 1 is smallest, n is largest (bottom initial)
    Each move: [disk_id, from_peg, to_peg]
    """
    if not isinstance(moves, list):
        return {"success": False, "illegal_step": None, "move_count": 0}

    pegs = {0: list(range(n, 0, -1)), 1: [], 2: []}  # top is last element

    for idx, mv in enumerate(moves):
        if not isinstance(mv, (list, tuple)) or len(mv) != 3:
            return {"success": False, "illegal_step": idx + 1, "move_count": idx + 1}

        disk, src, dst = mv

        # Type / value checks
        if not isinstance(disk, int) or not isinstance(src, int) or not isinstance(dst, int):
            return {"success": False, "illegal_step": idx + 1, "move_count": idx + 1}

        if disk < 1 or disk > n or src not in pegs or dst not in pegs:
            return {"success": False, "illegal_step": idx + 1, "move_count": idx + 1}

        # Source must have at least one disk and top must match disk
        if not pegs[src] or pegs[src][-1] != disk:
            return {"success": False, "illegal_step": idx + 1, "move_count": idx + 1}

        # Pop from src
        pegs[src].pop()

        # Destination rule: cannot place larger disk on smaller one
        if pegs[dst] and pegs[dst][-1] < disk:
            return {"success": False, "illegal_step": idx + 1, "move_count": idx + 1}

        pegs[dst].append(disk)

    success = (
        pegs[0] == [] and
        pegs[1] == [] and
        pegs[2] == list(range(n, 0, -1))
    )

    return {"success": success, "illegal_step": None, "move_count": len(moves)}


# -------- MAIN EXPERIMENT LOOP --------
def run_simple_experiment():
    per_size_rows = []

    for n in DISC_SIZES:
        print(f"\n=== Disc size n={n} ===")

        success_count = 0
        sample_moves_raw = None
        sample_move_count = None
        last_parse_error = None

        # token-usage accumulators
        sum_input_tokens = 0
        sum_output_tokens = 0
        sum_total_tokens = 0
        usage_count = 0

        for t in range(1, TRIALS_PER_SIZE + 1):
            max_new_tokens_n = min(2000 * n, 10000)
            # Add a random "seed tag" in user prompt so trials differ
            seed_tag = f"[trial_seed:{random.randint(0, 999999)}]"
            user_prompt = make_user_prompt(n) + "\n" + seed_tag

            print(f"\n--- Trial {t}/{TRIALS_PER_SIZE} for n={n} ---")
            print(f"Using max_new_tokens = {max_new_tokens_n}, seed_tag = {seed_tag}")

            text, usage = call_local_model(SYSTEM_PROMPT, user_prompt, max_new_tokens_n)
            moves, parse_error = extract_moves(text)
            validation = validate_hanoi(n, moves)

            trial_success = bool(validation["success"])
            if trial_success:
                success_count += 1
                if sample_moves_raw is None:
                    sample_moves_raw = repr(moves)
                    sample_move_count = validation.get("move_count")

            if usage:
                it = usage["input_tokens"]
                ot = usage["output_tokens"]
                tt = usage["total_tokens"]
                sum_input_tokens += it
                sum_output_tokens += ot
                sum_total_tokens += tt
                usage_count += 1
            else:
                it = ot = tt = None

            last_parse_error = parse_error

            # Per-trial logging
            print(f"Success: {trial_success}")
            print(f"Move count: {validation.get('move_count')}")
            print(f"Parse error: {parse_error}")
            print(f"Tokens - input: {it}, output: {ot}, total: {tt}")

            if parse_error:
                snippet = text[:250].replace("\n", " ")
                print(f"Output snippet: {snippet}...")

        accuracy = success_count / TRIALS_PER_SIZE if TRIALS_PER_SIZE > 0 else 0.0

        if usage_count > 0:
            avg_input_tokens = sum_input_tokens / usage_count
            avg_output_tokens = sum_output_tokens / usage_count
            avg_total_tokens = sum_total_tokens / usage_count
        else:
            avg_input_tokens = avg_output_tokens = avg_total_tokens = None

        per_size_rows.append({
            "disc_size": n,
            "trials": TRIALS_PER_SIZE,
            "successes": success_count,
            "accuracy": accuracy,
            "sample_move_count": sample_move_count,
            "expected_min_moves": 2**n - 1,
            "sample_moves_raw": sample_moves_raw,
            "avg_tokens": {
                "input": avg_input_tokens,
                "output": avg_output_tokens,
                "total": avg_total_tokens,
            },
            "last_parse_error": last_parse_error,
        })

        print(f"\n=== Summary for n={n} ===")
        print(f"Accuracy: {success_count}/{TRIALS_PER_SIZE} = {accuracy:.3f}")
        print(f"Avg output tokens: {avg_output_tokens}")
        print(f"Last parse error: {last_parse_error}")

    # Save JSON summary
    summary = {row["disc_size"]: row for row in per_size_rows}
    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary JSON to {OUT_SUMMARY}")
    print("Accuracies per disc size:")
    for r in per_size_rows:
        print(
            f"n={r['disc_size']}: {r['successes']}/{r['trials']} "
            f"= {r['accuracy']:.3f}, avg_output_tokens={r['avg_tokens']['output']}"
        )


if __name__ == "__main__":
    run_simple_experiment()