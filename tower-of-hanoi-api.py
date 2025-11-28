#!/usr/bin/env python3
"""
experiment_hanoi.py

Groq Tower-of-Hanoi experiment (0-indexed pegs, disk-id moves).

- System & user prompts are modeled after game1.py (DeepSeek local script):
  * 3 pegs, 0-indexed
  * disks numbered 1 (smallest) .. N (largest)
  * final answer format: moves = [[disk id, from peg, to peg], ...]
- Uses Groq ChatCompletion API (streaming).
- Runs TRIALS_PER_SIZE trials per disc size (no early abort).
- Accuracy for each problem size = (# successful trials) / TRIALS_PER_SIZE.
- Extracts `moves = [[...], ...]` via regex + ast.literal_eval.
- Validates moves by simulating Hanoi.
- Stores per-size CSV + JSON summary.
"""

import os
import json
import time
import csv
import re
import ast

from groq import Groq
from tqdm import tqdm

# -------- CONFIG --------
MODEL = "openai/gpt-oss-20b"   # change if needed
TEMPERATURE = 0.0
MAX_TOKENS = 10000

N_DISCS_LIST = [5, 16, 20]
TRIALS_PER_SIZE = 1        # total trials per size

OUT_CSV = "./W2/hanoi_results.csv"
OUT_SUMMARY = "./W2/hanoi_summary.json"

client = Groq()

# -------- SYSTEM PROMPT --------
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
- You may reason step-by-step in your explanation.
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
""".strip()


# -------- CALL GROQ --------
def call_groq_stream(system_prompt: str, user_prompt: str):
    """
    Calls Groq ChatCompletion API (streaming) and returns (text, usage_dict).
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_TOKENS,
        stream=True,
    )

    text = ""
    usage = None
    for chunk in stream:
        # accumulate text
        try:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                text += delta.content
            elif isinstance(delta, dict):
                c = delta.get("content")
                if c:
                    text += c
        except Exception:
            pass

        # usage metadata (often in last chunk)
        if hasattr(chunk, "usage") and getattr(chunk, "usage") is not None:
            raw = chunk.usage

            def _get(k):
                try:
                    if isinstance(raw, dict):
                        return raw.get(k)
                    return getattr(raw, k)
                except Exception:
                    return None

            input_t = _get("input_tokens") or _get("prompt_tokens") or _get(
                "input_token_count"
            ) or _get("prompt_token_count")
            output_t = _get("output_tokens") or _get("completion_tokens") or _get(
                "output_token_count"
            )
            total_t = _get("total_tokens") or _get("total_token_count")

            try:
                input_t = int(input_t) if input_t is not None else None
            except Exception:
                input_t = None
            try:
                output_t = int(output_t) if output_t is not None else None
            except Exception:
                output_t = None
            try:
                total_t = int(total_t) if total_t is not None else None
            except Exception:
                total_t = None

            if total_t is None and input_t is not None and output_t is not None:
                total_t = input_t + output_t

            usage = {
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": total_t,
            }

    return text.strip(), usage


# -------- PARSE moves = [[...], ...] --------
def extract_moves(text: str):
    """
    Extracts the Python-like list from an expression of the form:
       moves = [[disk, from peg, to peg], ...]
    Returns (moves_list, parse_error).
    """
    moves = None
    parse_error = None

    # Regex to find 'moves = [[...]]' capturing the list part
    m = re.search(r"moves\s*=\s*(\[\s*\[.*\]\s*\])", text, re.DOTALL)
    if not m:
        return None, "No 'moves = [[...]]' pattern found"

    list_str = m.group(1)
    try:
        # Safely evaluate the list literal
        moves = ast.literal_eval(list_str)
    except Exception as exc:
        parse_error = f"literal_eval error: {exc}"
        return None, parse_error

    # Basic sanity checks: list of length-3 lists/tuples
    if not isinstance(moves, list):
        return None, "Parsed 'moves' is not a list"

    for idx, mv in enumerate(moves):
        if not isinstance(mv, (list, tuple)) or len(mv) != 3:
            return None, f"Move {idx+1} is not a length-3 list"

    return moves, None


# -------- VALIDATE HANOI (unchanged from your script) --------
def validate_hanoi(n, moves):
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


# -------- MAIN EXPERIMENT LOOP (keeps your CSV/JSON structure) --------
def run_hanoi_experiment():
    per_size_rows = []
    detailed_rows = []
    total = len(N_DISCS_LIST) * TRIALS_PER_SIZE
    pbar = tqdm(total=total, desc="Running Hanoi trials")

    for n in N_DISCS_LIST:
        success_count = 0
        sample_success = {
            "moves_raw": None,
            "reasoning": None,  # we'll store raw output here
            "move_count": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }
        last_parse_error = None

        for t in range(1, TRIALS_PER_SIZE + 1):
            system = SYSTEM_PROMPT
            user = make_user_prompt(n)
            text, usage = call_groq_stream(system, user)

            # Use regex + literal_eval to get moves
            moves, parse_error = extract_moves(text)
            validation = validate_hanoi(n, moves)

            trial_success = bool(validation["success"])
            if trial_success:
                success_count += 1
                if sample_success["moves_raw"] is None:
                    sample_success["moves_raw"] = repr(moves)
                    sample_success["reasoning"] = text
                    sample_success["move_count"] = validation.get("move_count")
                    if usage:
                        sample_success["input_tokens"] = usage.get("input_tokens")
                        sample_success["output_tokens"] = usage.get("output_tokens")
                        sample_success["total_tokens"] = usage.get("total_tokens")

            input_tokens = usage.get("input_tokens") if usage else None
            output_tokens = usage.get("output_tokens") if usage else None
            total_tokens = usage.get("total_tokens") if usage else None

            detailed_rows.append({
                "n_discs": n,
                "trial": t,
                "success": int(trial_success),
                "parse_error": parse_error,
                "move_count": validation.get("move_count"),
                "reasoning": text,  # full raw output
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            })

            last_parse_error = parse_error
            pbar.update(1)
            time.sleep(0.1)

        accuracy = success_count / TRIALS_PER_SIZE

        size_trials = [dr for dr in detailed_rows if dr["n_discs"] == n]

        def avg(field):
            vals = [dr[field] for dr in size_trials if dr[field] is not None]
            return sum(vals) / len(vals) if vals else None

        avg_input_tokens = avg("input_tokens")
        avg_output_tokens = avg("output_tokens")
        avg_total_tokens = avg("total_tokens")

        per_size_rows.append({
            "n_discs": n,
            "trials": TRIALS_PER_SIZE,
            "successes": success_count,
            "accuracy": accuracy,
            "sample_move_count": sample_success["move_count"],
            "expected_min_moves": 2**n - 1,
            "sample_moves_raw": sample_success["moves_raw"],
            "sample_reasoning": sample_success["reasoning"],
            "sample_input_tokens": sample_success["input_tokens"],
            "sample_output_tokens": sample_success["output_tokens"],
            "sample_total_tokens": sample_success["total_tokens"],
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "avg_total_tokens": avg_total_tokens,
            "last_parse_error": last_parse_error,
        })

    pbar.close()

    # Write CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = [
        "n_discs", "trials", "successes", "accuracy",
        "sample_move_count", "expected_min_moves",
        "sample_moves_raw", "sample_reasoning",
        "sample_input_tokens", "sample_output_tokens", "sample_total_tokens",
        "avg_input_tokens", "avg_output_tokens", "avg_total_tokens",
        "last_parse_error",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_size_rows:
            writer.writerow(r)

    # JSON summary
    summary = {}
    for r in per_size_rows:
        summary[r["n_discs"]] = {
            "trials": r["trials"],
            "successes": r["successes"],
            "accuracy": r["accuracy"],
            "sample_move_count": r["sample_move_count"],
            "expected_min_moves": r["expected_min_moves"],
            "sample_tokens": {
                "input_tokens": r["sample_input_tokens"],
                "output_tokens": r["sample_output_tokens"],
                "total_tokens": r["sample_total_tokens"],
            },
            "avg_tokens": {
                "input_tokens": r["avg_input_tokens"],
                "output_tokens": r["avg_output_tokens"],
                "total_tokens": r["avg_total_tokens"],
            },
            "last_parse_error": r["last_parse_error"],
        }

    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved per-size CSV to {OUT_CSV}")
    print(f"Saved summary JSON to {OUT_SUMMARY}")
    print("Accuracies per n_discs:")
    for r in per_size_rows:
        print(f"n={r['n_discs']}: {r['successes']}/{r['trials']} = {r['accuracy']:.3f}")
        print(f"  avg_output_tokens = {r['avg_output_tokens']}")


if __name__ == "__main__":
    run_hanoi_experiment()
