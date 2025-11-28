#!/usr/bin/env python3
"""
experiment_checker_jumping.py

Groq Checker-Jumping experiment.

- Initial state: R R ... R _ B B ... B (N reds, 1 empty, N blues).
- Goal state:    B B ... B _ R R ... R
- Moves: [checker_color, position_from, position_to]
  with checker_color in {"R", "B"}.
- Sliding: to adjacent empty.
- Jump: over exactly one opposite-color checker into empty.
- Checkers cannot move backwards (towards starting side).

Same experiment structure as Hanoi:
- TRIALS_PER_SIZE trials per N.
- Accuracy, token stats, and a sample successful run.
"""

import os
import json
import time
import csv
from groq import Groq
from tqdm import tqdm

# -------- CONFIG --------
MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.0
MAX_TOKENS = 10000

N_CHECKERS_LIST = [4, 5, 6, 8, 10, 12, 16, 20]   # N = number of reds and blues
TRIALS_PER_SIZE = 1

OUT_CSV = "./W2/checkers_results.csv"
OUT_SUMMARY = "./W2/checkers_summary.json"

client = Groq()

# -------- PROMPT --------
def make_prompt(n):
    system = (
        "You are a helpful assistant. Solve this checker-jumping puzzle for me.\n\n"
        "On a one-dimensional board, there are red checkers ('R'), blue checkers ('B'), "
        "and one empty space ('_'). A checker can move by either:\n"
        "1. Sliding forward into an adjacent empty space, or\n"
        "2. Jumping over exactly one checker of the opposite color to land in an empty space.\n\n"
        "We use 0-indexed positions (leftmost position is 0).\n"
        "Checkers cannot move backwards (towards their starting side):\n"
        "- Red checkers can only move to the right (their position must strictly increase).\n"
        "- Blue checkers can only move to the left (their position must strictly decrease).\n\n"
        "The goal is to swap the positions of all red and blue checkers, effectively mirroring "
        "the initial state.\n"
        "Your solution should be a list of moves where each move is [checker_color, position_from, position_to].\n\n"
        "When exploring potential solutions in your thinking process, always include the corresponding "
        "complete list of moves.\n\n"
        "Your FINAL OUTPUT MUST HAVE EXACTLY TWO MARKED SECTIONS:\n"
        "1) <REASONING> ... </REASONING> — your chain-of-thought reasoning (may include sequences of moves).\n"
        "2) <ANSWER> ... </ANSWER>      — a valid JSON object ONLY of the form:\n"
        "   {\"moves\": [[checker_color, position_from, position_to], ...]}\n"
        "   - checker_color must be \"R\" or \"B\".\n"
        "   - position_from and position_to must be integers.\n"
        "DO NOT output anything outside these two marked sections.\n"
    )

    user = (
        f"I have a puzzle with 2*{n}+1 positions.\n"
        f"There are {n} red checkers ('R') on the left, {n} blue checkers ('B') on the right, "
        "and one empty space ('_') in the middle.\n\n"
        "Initial board (0-indexed): R R ... R _ B B ... B\n"
        "Goal board:                B B ... B _ R R ... R\n\n"
        "Rules:\n"
        "• A checker can slide into an adjacent empty space.\n"
        "• A checker can jump over exactly one checker of the opposite color to land in an empty space.\n"
        "• Red checkers move only to the right.\n"
        "• Blue checkers move only to the left.\n"
        "• No backwards moves.\n\n"
        "Find the minimum sequence of moves (or at least a valid sequence) to transform the initial board "
        "into the goal board.\n\n"
        "<REASONING>\n</REASONING>\n\n"
        "<ANSWER>{\"moves\": []}</ANSWER>\n"
    )

    return system, user

# -------- CALL & STREAM --------
def call_groq_stream(system_prompt, user_prompt):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_TOKENS,
        stream=True
    )

    text = ""
    usage = None
    for chunk in stream:
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

        if hasattr(chunk, "usage") and getattr(chunk, "usage") is not None:
            raw = chunk.usage

            def _get(k):
                try:
                    if isinstance(raw, dict):
                        return raw.get(k)
                    return getattr(raw, k)
                except Exception:
                    return None

            input_t = _get("input_tokens") or _get("prompt_tokens") or _get("input_token_count") or _get("prompt_token_count")
            output_t = _get("output_tokens") or _get("completion_tokens") or _get("output_token_count")
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

    return text, usage

# -------- PARSE SECTIONS & JSON --------
def extract_reasoning_and_moves(text):
    def between(s, a, b):
        lo = s.lower()
        a_lo = a.lower()
        b_lo = b.lower()
        i = lo.find(a_lo)
        if i == -1:
            return None
        j = lo.find(b_lo, i + len(a_lo))
        if j == -1:
            return None
        return s[i + len(a): j].strip()

    reasoning = between(text, "<REASONING>", "</REASONING>")
    answer_block = between(text, "<ANSWER>", "</ANSWER>")

    moves = None
    parse_error = None
    if answer_block:
        s = answer_block.find("{")
        e = answer_block.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(answer_block[s:e+1])
                moves = data.get("moves")
            except Exception as exc:
                parse_error = f"JSON parse error: {exc}"
        else:
            parse_error = "No JSON object found inside <ANSWER> block"
    else:
        parse_error = "No <ANSWER> block found"

    return reasoning, moves, parse_error

# -------- VALIDATE CHECKER JUMPING --------
def validate_checkers(n, moves):
    """
    State is a list of length 2n+1:
    [ 'R', ..., 'R', '_', 'B', ..., 'B' ]
    Goal: [ 'B', ..., 'B', '_', 'R', ..., 'R' ]
    Moves: [color, from_pos, to_pos]
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

        if color not in ("R", "B"):
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        if not (in_bounds(src) and in_bounds(dst)):
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        if board[src] != color:
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        if board[dst] != '_':
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        # Direction constraint
        if color == 'R':
            if dst <= src:
                return {"success": False, "illegal_step": step_no, "move_count": step_no}
        else:  # 'B'
            if dst >= src:
                return {"success": False, "illegal_step": step_no, "move_count": step_no}

        # Slide vs jump
        dist = abs(dst - src)
        if dist == 1:
            # simple slide
            pass
        elif dist == 2:
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

# -------- MAIN LOOP --------
def run_checker_experiment():
    per_size_rows = []
    detailed_rows = []
    total = len(N_CHECKERS_LIST) * TRIALS_PER_SIZE
    pbar = tqdm(total=total, desc="Running checker-jumping trials")

    for n in N_CHECKERS_LIST:
        success_count = 0
        sample_success = {
            "moves_raw": None,
            "reasoning": None,
            "move_count": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }
        last_parse_error = None

        for t in range(1, TRIALS_PER_SIZE + 1):
            system, user = make_prompt(n)
            text, usage = call_groq_stream(system, user)
            reasoning, moves, parse_error = extract_reasoning_and_moves(text)
            validation = validate_checkers(n, moves)

            trial_success = bool(validation["success"])
            if trial_success:
                success_count += 1
                if sample_success["moves_raw"] is None:
                    sample_success["moves_raw"] = json.dumps(moves)
                    sample_success["reasoning"] = reasoning
                    sample_success["move_count"] = validation.get("move_count")
                    if usage:
                        sample_success["input_tokens"] = usage.get("input_tokens")
                        sample_success["output_tokens"] = usage.get("output_tokens")
                        sample_success["total_tokens"] = usage.get("total_tokens")

            input_tokens = usage.get("input_tokens") if usage else None
            output_tokens = usage.get("output_tokens") if usage else None
            total_tokens = usage.get("total_tokens") if usage else None

            detailed_rows.append({
                "n_checkers": n,
                "trial": t,
                "success": int(trial_success),
                "parse_error": parse_error,
                "move_count": validation.get("move_count"),
                "reasoning": reasoning,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            })

            last_parse_error = parse_error
            pbar.update(1)
            time.sleep(0.2)

        accuracy = success_count / TRIALS_PER_SIZE

        size_trials = [dr for dr in detailed_rows if dr["n_checkers"] == n]

        def avg(field):
            vals = [dr[field] for dr in size_trials if dr[field] is not None]
            return sum(vals) / len(vals) if vals else None

        avg_input_tokens = avg("input_tokens")
        avg_output_tokens = avg("output_tokens")
        avg_total_tokens = avg("total_tokens")

        # Known minimal move count for this puzzle: N^2 + 2N
        expected_min_moves = n * n + 2 * n

        per_size_rows.append({
            "n_checkers": n,
            "trials": TRIALS_PER_SIZE,
            "successes": success_count,
            "accuracy": accuracy,
            "sample_move_count": sample_success["move_count"],
            "expected_min_moves": expected_min_moves,
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

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = [
        "n_checkers", "trials", "successes", "accuracy",
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

    summary = {}
    for r in per_size_rows:
        summary[r["n_checkers"]] = {
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
    print("Accuracies per n_checkers:")
    for r in per_size_rows:
        print(f"N={r['n_checkers']}: {r['successes']}/{r['trials']} = {r['accuracy']:.3f}")
        print(f"  avg_output_tokens = {r['avg_output_tokens']}")

if __name__ == "__main__":
    run_checker_experiment()
