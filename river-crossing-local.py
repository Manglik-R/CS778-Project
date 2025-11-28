#!/usr/bin/env python3
"""
River-crossing experiment on a LOCAL DeepSeek-R1 model.

- N actors a_1..a_N and N agents A_1..A_N.
- Boat capacity k.
- Initially, all on LEFT bank with the boat.
- Each move is a boat crossing: ["A_2", "a_2"], ["A_2"], ...

Constraint:
- No actor may ever be together with another agent (on either bank or in the boat),
  unless that actor's own agent is also present.
  Equivalently: For any group (left bank, right bank, or boat), if there is an actor a_i
  and some agent A_j with j != i, then A_i must also be present.

We:
- Prompt a local DeepSeek-R1 model (no external API).
- Ask for final answer in the exact format:
    moves = [["A_2", "a_2"], ["A_2"], ...]
- Parse with regex + ast.literal_eval.
- Validate sequences with a simulator that enforces the constraint and boat capacity.
- Run multiple (N, k) configurations and trials.
- Store a JSON summary (no CSV).
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

RIVER_PROBLEMS = [(2, 2), (4, 2), (6, 3)]
TRIALS_PER_PROBLEM = 5

OUT_SUMMARY = "./game3/summary.json"

# -------- SYSTEM PROMPT (river-crossing spec) --------
SYSTEM_PROMPT = """
You are a helpful assistant. Solve this river-crossing puzzle for me.

There are N actors and their N agents who want to cross a river in a boat.
Actors are denoted a_1, a_2, ..., and agents are denoted A_1, A_2, ...

The boat can hold at most k people at a time and cannot travel empty.

Constraint:
No actor can ever be in the presence of another agent (including while in the boat),
unless their own agent is also present in that same location.
Equivalently: For any group (left bank, right bank, or boat), if there is an actor a_i
and some agent A_j with j != i, then A_i must also be present in that group.

Initially, all actors and agents are on the LEFT bank with the boat.
The goal is to move everyone to the RIGHT bank.

A solution is a list of boat moves, where each move is a list of the people on the boat
for that crossing, e.g. ["A_2", "a_2"], ["A_2"], ...

Your final answer MUST be a single assignment of the form:
  moves = [["A_2", "a_2"], ["A_2"], ...]

Where:
- Each inner list is the (non-empty) group of people on the boat for that crossing.
- Names must be strings like "a_1", "A_1", "a_2", "A_2", ..., up to N.
- Each move respects all rules above (capacity and safety constraints).
"""

# -------- USER PROMPT TEMPLATE --------
def make_user_prompt(n: int, k: int) -> str:
    actors = ", ".join(f"a_{i}" for i in range(1, n + 1))
    agents = ", ".join(f"A_{i}" for i in range(1, n + 1))
    return f"""
We have a river-crossing puzzle with:

- {n} actors: {actors}
- {n} agents: {agents}

Initially:
- All actors and agents are on the LEFT bank of the river.
- The boat is also on the LEFT bank.

Goal:
- Move everyone to the RIGHT bank of the river.

Boat:
- The boat can carry at most {k} people at a time.
- The boat cannot travel empty (each crossing must have at least one passenger).

Safety constraint (very important):
- No actor can be in the presence of another agent (including while riding the boat),
  unless their own agent is also present in that same location.
  Equivalently: For any group (left bank, right bank, or boat), if there is an actor a_i
  and some agent A_j with j != i, then A_i must also be present in that group.

Find a valid sequence of boat crossings that moves everyone from the LEFT bank to the RIGHT bank
without ever violating the constraint and obeying the boat capacity.

Return your final answer in the EXACT format:
moves = [["A_2", "a_2"], ["A_2"], ...]
(Use only names of the form "a_i" or "A_i". The order of people within each list does not matter.)
""".strip()


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
            do_sample=True,
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
       moves = [["A_2", "a_2"], ["A_2"], ...]
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
        moves = ast.literal_eval(list_str)
    except Exception as exc:
        parse_error = f"literal_eval error: {exc}"
        return None, parse_error

    if not isinstance(moves, list):
        return None, "Parsed 'moves' is not a list"

    for idx, mv in enumerate(moves):
        if not isinstance(mv, (list, tuple)) or len(mv) == 0:
            return None, f"Move {idx+1} is not a non-empty list"
        for person in mv:
            if not isinstance(person, str):
                return None, f"Move {idx+1} contains a non-string name"

    return moves, None


# -------- VALIDATION HELPERS (copied from your Groq script) --------
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
        # unknown names handled elsewhere

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
    for i in range(1, n + 1):
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


# -------- MAIN EXPERIMENT LOOP --------
def run_river_experiment():
    per_problem_rows = []

    for (n, k) in RIVER_PROBLEMS:
        print(f"\n=== Problem: N={n} pairs, boat capacity k={k} ===")

        success_count = 0
        sample_moves_raw = None
        sample_move_count = None
        last_parse_error = None

        # token-usage accumulators
        sum_input_tokens = 0
        sum_output_tokens = 0
        sum_total_tokens = 0
        usage_count = 0

        for t in range(1, TRIALS_PER_PROBLEM + 1):
            # Scale with N, upper bound 10k
            max_new_tokens_n = min(2000 * 5, 10000)

            seed_tag = f"[trial_seed:{random.randint(0, 999999)}]"
            user_prompt = make_user_prompt(n, k) + "\n" + seed_tag

            print(f"\n--- Trial {t}/{TRIALS_PER_PROBLEM} for N={n}, k={k} ---")
            print(f"Using max_new_tokens = {max_new_tokens_n}, seed_tag = {seed_tag}")

            text, usage = call_local_model(SYSTEM_PROMPT, user_prompt, max_new_tokens_n)
            moves, parse_error = extract_moves(text)
            validation = validate_river(n, k, moves)

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

        accuracy = success_count / TRIALS_PER_PROBLEM if TRIALS_PER_PROBLEM > 0 else 0.0

        if usage_count > 0:
            avg_input_tokens = sum_input_tokens / usage_count
            avg_output_tokens = sum_output_tokens / usage_count
            avg_total_tokens = sum_total_tokens / usage_count
        else:
            avg_input_tokens = avg_output_tokens = avg_total_tokens = None

        per_problem_rows.append({
            "n_pairs": n,
            "boat_capacity": k,
            "trials": TRIALS_PER_PROBLEM,
            "successes": success_count,
            "accuracy": accuracy,
            "sample_move_count": sample_move_count,
            "expected_min_moves": None,  # not easily closed-form
            "sample_moves_raw": sample_moves_raw,
            "avg_tokens": {
                "input": avg_input_tokens,
                "output": avg_output_tokens,
                "total": avg_total_tokens,
            },
            "last_parse_error": last_parse_error,
        })

        print(f"\n=== Summary for N={n}, k={k} ===")
        print(f"Accuracy: {success_count}/{TRIALS_PER_PROBLEM} = {accuracy:.3f}")
        print(f"Avg output tokens: {avg_output_tokens}")
        print(f"Last parse error: {last_parse_error}")

    # Save JSON summary
    summary = {
        f"N={row['n_pairs']},k={row['boat_capacity']}": row
        for row in per_problem_rows
    }
    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary JSON to {OUT_SUMMARY}")
    print("Accuracies per (n_pairs, boat_capacity):")
    for r in per_problem_rows:
        print(
            f"N={r['n_pairs']},k={r['boat_capacity']}: "
            f"{r['successes']}/{r['trials']} = {r['accuracy']:.3f}, "
            f"avg_output_tokens={r['avg_tokens']['output']}"
        )


if __name__ == "__main__":
    run_river_experiment()