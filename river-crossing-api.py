#!/usr/bin/env python3
"""
experiment_river_crossing.py

Groq River-Crossing experiment.

- N actors a_1..a_N and N agents A_1..A_N.
- Boat capacity k.
- Initially, all on left with boat.
- Moves: each move is a list of people on the boat,
  e.g. ["A_2", "a_2"], ["A_2"], ...

Constraint:
- No actor may ever be together with another agent (on either bank or in the boat),
  unless that actor's own agent is also present.

We simulate moves, enforce constraints at every step (including boat group),
for multiple (N, k) configurations.

Output format in <ANSWER>:
{ "moves": [["A_2", "a_2"], ["A_2"], ...] }
"""

import os
import json
import time
import csv
from groq import Groq
from tqdm import tqdm

# -------- CONFIG --------
MODEL = "openai/gpt-oss-120b"   # change if needed
TEMPERATURE = 0.0
MAX_TOKENS = 10000

RIVER_PROBLEMS = [(2, 2)]
TRIALS_PER_PROBLEM = 1

OUT_CSV = "./W2/river_results.csv"
OUT_SUMMARY = "./W2/river_summary.json"

client = Groq()

# -------- PROMPT --------
def make_prompt(n, k):
    system = (
        "You are a helpful assistant. Solve this river-crossing puzzle for me.\n\n"
        "There are N actors and their N agents who want to cross a river in a boat. "
        "Actors are denoted a_1, a_2, ..., and agents are denoted A_1, A_2, ...\n"
        f"The boat can hold at most {k} people at a time and cannot travel empty.\n\n"
        "Constraint:\n"
        "No actor can ever be in the presence of another agent (including while in the boat), "
        "unless their own agent is also present in that same location.\n"
        "Equivalently: For any group (left bank, right bank, or boat), if there is an actor a_i "
        "and some agent A_j with j != i, then A_i must also be present in that group.\n\n"
        "A solution is a list of boat moves, where each move is a list of people on the boat "
        "for that crossing, e.g. [\"A_2\", \"a_2\"], [\"A_2\"], ...\n\n"
        "When exploring potential solutions in your thinking process, always include the "
        "corresponding complete list of boat moves.\n\n"
        "Your FINAL OUTPUT MUST HAVE EXACTLY TWO MARKED SECTIONS:\n"
        "1) <REASONING> ... </REASONING> — your chain-of-thought reasoning.\n"
        "2) <ANSWER> ... </ANSWER>      — a valid JSON object ONLY of the form:\n"
        "   {\"moves\": [[\"A_2\", \"a_2\"], [\"A_2\"], ...]}\n"
        "   - Each inner list is the group of people on the boat for that crossing.\n"
        "   - Names must be strings like \"a_1\", \"A_1\", \"a_2\", \"A_2\", ..., up to N.\n"
        "DO NOT output anything outside these two marked sections.\n"
    )

    actors = ", ".join(f"a_{i}" for i in range(1, n+1))
    agents = ", ".join(f"A_{i}" for i in range(1, n+1))

    user = (
        f"There are {n} actors and their {n} agents.\n"
        f"Actors: {actors}\n"
        f"Agents: {agents}\n\n"
        "Initially, all actors and agents are on the LEFT side of the river with the boat.\n"
        "Goal: move everyone to the RIGHT side of the river.\n\n"
        f"The boat can hold only {k} people at a time, and it cannot travel empty.\n\n"
        "Constraint (very important):\n"
        "No actor can be in the presence of another agent, including while riding the boat, "
        "unless their own agent is also present.\n\n"
        "Find a valid sequence of boat crossings that moves everyone from the left side "
        "to the right side without ever violating the constraint.\n\n"
        "Your answer must be in the format:\n"
        "  moves = [[\"A_2\", \"a_2\"], [\"A_2\"], ...]\n\n"
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

# -------- VALIDATE RIVER CROSSING --------
def _parse_person_name(name):
    """
    Returns (kind, idx) where kind in {"actor", "agent"} and idx is int, or (None, None) on error.
    actor: "a_i", agent: "A_i"
    """
    if not isinstance(name, str):
        return None, None
    name = name.strip()
    if len(name) < 3 or name[1] != "_":
        return None, None
    prefix = name[0]
    try:
        idx = int(name[2:])
    except Exception:
        return None, None

    if prefix == "a":
        return "actor", idx
    if prefix == "A":
        return "agent", idx
    return None, None

def _safe_group(group):
    """
    group is an iterable of person names.
    Constraint: For any actor a_i with some agent A_j (j != i) in the same group,
    A_i must also be present.
    """
    actors = set()
    agents = set()
    for name in group:
        kind, idx = _parse_person_name(name)
        if kind == "actor":
            actors.add(idx)
        elif kind == "agent":
            agents.add(idx)
        # unknown names are treated as invalid elsewhere

    for i in actors:
        # if there exists some agent j != i
        if any(j != i for j in agents):
            # then A_i must also be present
            if i not in agents:
                return False
    return True

def validate_river(n, k, moves):
    """
    Validate river crossing with N actor/agent pairs and boat capacity k.
    moves: list of lists (people on boat each crossing).
    """
    if not isinstance(moves, list):
        return {"success": False, "illegal_step": None, "move_count": 0}

    # Initial sets
    all_people = set()
    for i in range(1, n+1):
        all_people.add(f"a_{i}")
        all_people.add(f"A_{i}")

    left = set(all_people)
    right = set()
    boat_side = "L"  # "L" or "R"

    # Sanity: initial configuration must be safe
    if not _safe_group(left) or not _safe_group(right):
        return {"success": False, "illegal_step": 0, "move_count": 0}

    for idx, mv in enumerate(moves):
        step_no = idx + 1
        # mv should be a list of names, 1..k people
        if not isinstance(mv, (list, tuple)) or len(mv) == 0 or len(mv) > k:
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        mv_set = set(mv)

        # All names must be valid and present on current boat side
        side = left if boat_side == "L" else right
        if not mv_set.issubset(side):
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        for name in mv_set:
            kind, idx_person = _parse_person_name(name)
            if kind is None or idx_person < 1 or idx_person > n:
                return {"success": False, "illegal_step": step_no, "move_count": step_no}

        # Boat cannot travel empty (already ensured by len(mv_set) > 0)

        # Check constraint on boat group
        if not _safe_group(mv_set):
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

        # Apply crossing
        if boat_side == "L":
            left = left - mv_set
            right = right | mv_set
            boat_side = "R"
        else:
            right = right - mv_set
            left = left | mv_set
            boat_side = "L"

        # Check constraint on both banks after crossing
        if not _safe_group(left) or not _safe_group(right):
            return {"success": False, "illegal_step": step_no, "move_count": step_no}

    # Success if everyone on right, left empty
    success = (len(left) == 0 and right == all_people)
    return {"success": success, "illegal_step": None, "move_count": len(moves)}

# -------- MAIN LOOP --------
def run_river_experiment():
    per_problem_rows = []
    detailed_rows = []
    total = len(RIVER_PROBLEMS) * TRIALS_PER_PROBLEM
    pbar = tqdm(total=total, desc="Running river-crossing trials")

    for (n, k) in RIVER_PROBLEMS:
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

        for t in range(1, TRIALS_PER_PROBLEM + 1):
            system, user = make_prompt(n, k)
            text, usage = call_groq_stream(system, user)
            reasoning, moves, parse_error = extract_reasoning_and_moves(text)
            validation = validate_river(n, k, moves)

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
                "n_pairs": n,
                "boat_capacity": k,
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

        accuracy = success_count / TRIALS_PER_PROBLEM

        problem_trials = [
            dr for dr in detailed_rows
            if dr["n_pairs"] == n and dr["boat_capacity"] == k
        ]

        def avg(field):
            vals = [dr[field] for dr in problem_trials if dr[field] is not None]
            return sum(vals) / len(vals) if vals else None

        avg_input_tokens = avg("input_tokens")
        avg_output_tokens = avg("output_tokens")
        avg_total_tokens = avg("total_tokens")

        per_problem_rows.append({
            "n_pairs": n,
            "boat_capacity": k,
            "trials": TRIALS_PER_PROBLEM,
            "successes": success_count,
            "accuracy": accuracy,
            "sample_move_count": sample_success["move_count"],
            "expected_min_moves": None,  # not easily closed-form; leave as None
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
        "n_pairs", "boat_capacity", "trials", "successes", "accuracy",
        "sample_move_count", "expected_min_moves",
        "sample_moves_raw", "sample_reasoning",
        "sample_input_tokens", "sample_output_tokens", "sample_total_tokens",
        "avg_input_tokens", "avg_output_tokens", "avg_total_tokens",
        "last_parse_error",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_problem_rows:
            writer.writerow(r)

    summary = {}
    for r in per_problem_rows:
        key = f"N={r['n_pairs']},k={r['boat_capacity']}"
        summary[key] = {
            "n_pairs": r["n_pairs"],
            "boat_capacity": r["boat_capacity"],
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

    print(f"Saved per-problem CSV to {OUT_CSV}")
    print(f"Saved summary JSON to {OUT_SUMMARY}")
    print("Accuracies per (n_pairs, boat_capacity):")
    for r in per_problem_rows:
        print(f"N={r['n_pairs']}, k={r['boat_capacity']}: "
              f"{r['successes']}/{r['trials']} = {r['accuracy']:.3f}")
        print(f"  avg_output_tokens = {r['avg_output_tokens']}")

if __name__ == "__main__":
    run_river_experiment()