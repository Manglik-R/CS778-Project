#!/usr/bin/env python3
"""

Checker-jumping experiment on a LOCAL DeepSeek-R1 model.

- System & user prompts specify:
  * 1D board of length 2N+1
  * N red checkers ('R') on the left
  * One empty space ('_') in the middle
  * N blue checkers ('B') on the right
  * final answer format: moves = [[checker_color, position_from, position_to], ...]
- Uses local Hugging Face model (no external API).
- max_new_tokens per trial = min(2000 * N, 10000).
- Uses sampling (temperature/top_p) + random seed tag so each trial is different.
- Validates the moves by simulating checker-jumping rules.
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

N_CHECKERS_LIST = [1, 2, 3]
TRIALS_PER_SIZE = 5

OUT_SUMMARY = "./game2/summary.json"

# -------- SYSTEM PROMPT (checker-jumping spec) --------
SYSTEM_PROMPT = """
You are a helpful assistant. Solve this checker-jumping puzzle for me.

On a one-dimensional board, there are red checkers ('R'), blue checkers ('B'),
and one empty space ('_'). A checker can move by either:
1. Sliding forward into an adjacent empty space, or
2. Jumping over exactly one checker of the opposite color to land in an empty space.

We use 0-indexed positions (leftmost position is 0).

Checkers cannot move backwards (towards their starting side):
- Red checkers can only move to the right (their position must strictly increase).
- Blue checkers can only move to the left (their position must strictly decrease).

The initial board has N red checkers on the left, then one empty space, then N blue checkers:
  [ 'R', ..., 'R', '_', 'B', ..., 'B' ]

The goal is to swap the positions of all red and blue checkers:
  [ 'B', ..., 'B', '_', 'R', ..., 'R' ]

Your final answer MUST be a single assignment of the form:
  moves = [[checker_color, position_from, position_to], ...]

Where:
- checker_color is "R" or "B"
- position_from and position_to are integers (0-indexed)
- Each move respects all rules above.
"""

# -------- USER PROMPT TEMPLATE --------
def make_user_prompt(n: int) -> str:
    size = 2 * n + 1
    return f"""
I have a checker-jumping puzzle with 2*{n}+1 = {size} positions.

Initial configuration (0-indexed):
- Positions 0..{n-1}: {n} red checkers 'R'
- Position {n}: one empty space '_'
- Positions {n+1}..{2*n}: {n} blue checkers 'B'

So the initial board is:
  R R ... R _ B B ... B

Goal configuration:
  B B ... B _ R R ... R

Rules:
- A checker can slide into an adjacent empty space.
- A checker can jump over exactly one checker of the opposite color into an empty space.
- Red checkers move only to the right (their index must strictly increase).
- Blue checkers move only to the left (their index must strictly decrease).
- No backwards moves and no jumping over more than one checker.

Find a sequence of moves to transform the initial configuration into the goal configuration.

Return your final answer in the EXACT format:
moves = [[checker_color, position_from, position_to], ...]
(Use only "R" or "B" for checker_color, and integers for positions.)
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
       moves = [[color, from, to], ...]
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


# -------- VALIDATE CHECKER-JUMPING SEQUENCE --------
def validate_checkers(n: int, moves):
    """
    State is a list of length 2n+1:
      [ 'R', ..., 'R', '_', 'B', ..., 'B' ]
    Goal: [ 'B', ..., 'B', '_', 'R', ..., 'R' ]
    Each move: [color, from_pos, to_pos]
    """
    if not isinstance(moves, list):
        return {"success": False, "illegal_step": None, "move_count": 0}

    size = 2 * n + 1
    board = ['R'] * n + ['_'] + ['B'] * n

    def in_bounds(pos):
        return isinstance(pos, int) and 0 <= pos < size

    for idx, mv in enumerate(moves):
        step_no = idx + 1

        if not (isinstance(mv, (list, tuple)) and len(mv) == 3):
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        color, src, dst = mv

        # Color must be "R" or "B"
        if not isinstance(color, str) or color not in ("R", "B"):
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        # Positions must be in-bounds integers
        if not (in_bounds(src) and in_bounds(dst)):
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        # Source must contain the correct checker
        if board[src] != color:
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        # Destination must be empty
        if board[dst] != '_':
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        # Direction constraint
        if color == 'R':
            if dst <= src:   # must move strictly right
                return {"success": False, "illegal_step": step_no, "move_count": step_no}
        else:  # 'B'
            if dst >= src:   # must move strictly left
                return {"success": False, "illegal_step": step_no, "move_count": step_no}

        # Slide vs jump
        dist = abs(dst - src)
        if dist == 1:
            # simple slide
            pass
        elif dist == 2:
            # jump over exactly one opposite-color checker
            mid = (src + dst) // 2
            if color == 'R' and board[mid] != 'B':
                return {"success": False, "illegal_step": step_no, "move_count": step_no}
            if color == 'B' and board[mid] != 'R':
                return {"success": False, "illegal_step": step_no, "move_count": step_no}
        else:
            # cannot jump over more than one checker
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        # Apply move
        board[dst] = color
        board[src] = '_'

    goal = ['B'] * n + ['_'] + ['R'] * n
    success = (board == goal)
    return {"success": success, "illegal_step": None, "move_count": len(moves)}


# -------- MAIN EXPERIMENT LOOP --------
def run_checker_experiment():
    per_size_rows = []

    for n in N_CHECKERS_LIST:
        print(f"\n=== Checker count per color n={n} ===")

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
            max_new_tokens_n = min(2000 * 5, 10000)
            # Add a random "seed tag" in user prompt so trials differ
            seed_tag = f"[trial_seed:{random.randint(0, 999999)}]"
            user_prompt = make_user_prompt(n) + "\n" + seed_tag

            print(f"\n--- Trial {t}/{TRIALS_PER_SIZE} for n={n} ---")
            print(f"Using max_new_tokens = {max_new_tokens_n}, seed_tag = {seed_tag}")

            text, usage = call_local_model(SYSTEM_PROMPT, user_prompt, max_new_tokens_n)
            moves, parse_error = extract_moves(text)
            validation = validate_checkers(n, moves)

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

        # Known minimal move count for this puzzle: N^2 + 2N
        expected_min_moves = n * n + 2 * n

        per_size_rows.append({
            "n_checkers": n,
            "trials": TRIALS_PER_SIZE,
            "successes": success_count,
            "accuracy": accuracy,
            "sample_move_count": sample_move_count,
            "expected_min_moves": expected_min_moves,
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
    summary = {row["n_checkers"]: row for row in per_size_rows}
    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary JSON to {OUT_SUMMARY}")
    print("Accuracies per n_checkers:")
    for r in per_size_rows:
        print(
            f"n={r['n_checkers']}: {r['successes']}/{r['trials']} "
            f"= {r['accuracy']:.3f}, avg_output_tokens={r['avg_tokens']['output']}"
        )


if __name__ == "__main__":
    run_checker_experiment()