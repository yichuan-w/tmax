"""Analyze generated tasks and solutions — summary tables and plots."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Complexity shortening ------------------------------------------------

# NOTE: "intricate" is the v2-only fourth complexity bucket added in commit
# ad9d7fe. Mirrors rl_data/generator/task_template_gen.py::TASK_COMPLEXITY.
_TASK_COMPLEXITY_ORDER = ["short", "moderate", "complex", "intricate"]
_CMD_COMPLEXITY_ORDER = ["bash-only", "bash+code", "bash+code+services"]

_CMD_COMPLEXITY_MAP = {
    "bash-only": "bash-only",
    "bash and code": "bash+code",
    "bash, code, and system services": "bash+code+services",
}

# ---- v2 axes (added in commit ad9d7fe — Verifier × Fixture × Corpus) -------
# Mirrors VERIFIER_KINDS / FIXTURE_KINDS / CORPUS_KINDS in
# rl_data.generator.task_template_gen. We pin a fixed display order so that
# legacy-only corpora and v2 corpora produce visually-comparable plots.
_VERIFIER_KIND_ORDER = [
    "exact_text",
    "metric_threshold",
    "adversarial_corpus",
    "fuzz_equivalence",
    "multi_protocol",
]
_FIXTURE_KIND_ORDER = [
    "text_only",
    "image",
    "audio",
    "video",
    "stripped_binary",
    "vendored_package",
    "multi_service_compose",
]
_CORPUS_KIND_ORDER = ["legacy", "sft_v2", "rl_v2"]
_BASE_IMAGE_ORDER = ["per_domain", "intricate"]

# Solver harness suffixes that ``rl_data.generate_solutions._summary_basename``
# may append to the per-task summary filename. Bash is the legacy / default
# and writes ``<MODEL_TAG>_summary.json`` (no suffix). Add to this tuple when
# new harnesses get wired into ``generate_solutions.py``.
_KNOWN_HARNESSES: tuple[str, ...] = ("vanillux",)


def _shorten_task_complexity(raw: str) -> str:
    m = re.match(r"(short|moderate|complex|intricate)\b", raw, re.IGNORECASE)
    return m.group(1).lower() if m else raw


def _shorten_cmd_complexity(raw: str) -> str:
    prefix = raw.split("(")[0].strip()
    return _CMD_COMPLEXITY_MAP.get(prefix, prefix)


def _resolve_pass_k(
    records: List[Dict[str, Any]],
    override: Optional[int] = None,
) -> int:
    """Return the canonical "ceiling" k to report alongside pass@1.

    The data-gen pipeline runs ``NUM_SOLUTIONS`` attempts per task and the
    summary stores ``pass_at_k`` as a dense dict keyed 1..NUM_SOLUTIONS. That
    means a corpus where some tasks were sampled at NUM_SOLUTIONS=4 (e.g. the
    vanillux smoke run) and others at NUM_SOLUTIONS=8 will have a *ragged*
    set of available k's. To stay apples-to-apples in cross-task aggregates,
    we pick the **largest k that every solved task in the corpus has data
    for** — i.e. the min of per-task max-k over solved records.

    *override*: when set (via ``--pass-at-k``), use it directly. Tasks that
    don't have that k get treated as missing in the aggregate (same handling
    as the previous hardcoded pass@8 path) but we warn so it's noticed.
    """
    if override is not None:
        return int(override)
    ks_max = [
        r.get("pass_k_max_avail", 0)
        for r in records
        if r.get("has_solutions") and r.get("pass_k_max_avail", 0) > 0
    ]
    if not ks_max:
        return 0
    return min(ks_max)


def _split_harness(slug: str) -> tuple[str, str]:
    """Split a slug discovered from a ``*_summary.json`` filename into
    ``(model_slug, harness)``.

    The naming convention (see ``rl_data.generate_solutions._summary_basename``)
    is:
    * ``<MODEL_TAG>_summary.json``               → harness = ``"bash"``  (legacy)
    * ``<MODEL_TAG>_<HARNESS>_summary.json``     → harness = ``<HARNESS>``

    We only strip *known* harness suffixes so model slugs that happen to end
    in things like ``_preview`` keep working unchanged.
    """
    for h in _KNOWN_HARNESSES:
        suffix = f"_{h}"
        if slug.endswith(suffix):
            return slug[: -len(suffix)], h
    return slug, "bash"


def _is_task_dir(p: Path) -> bool:
    """Predicate that recognizes both native `task_*` dirs and adapter-
    produced dirs (``otrl_task_*``, ``otb_*``, ...). Mirrors the one used by
    ``rl_data.generate_solutions`` and ``rl_data.comparison.taxonomy_classifier``."""
    return p.is_dir() and (p.name.startswith("task_") or (p / "task.json").exists())


def discover_models(tasks_dir: Path) -> List[tuple[str, str]]:
    """Return sorted list of ``(model_slug, harness)`` pairs found across all
    task solution dirs.

    Vanillux (and any other future non-bash harness) writes summaries as
    ``<MODEL>_<HARNESS>_summary.json`` so that bash and vanillux runs against
    the same model can coexist in one task dir. We surface each (model,
    harness) pair as its own analysis target so the per-axis plots stay
    apples-to-apples within a harness.
    """
    pairs: set[tuple[str, str]] = set()
    for task_path in tasks_dir.iterdir():
        if not _is_task_dir(task_path):
            continue
        solutions_dir = task_path / "solutions"
        if not solutions_dir.exists():
            continue
        for f in solutions_dir.glob("*_summary.json"):
            if f.name == "summary.json":
                continue
            raw_slug = f.name.removesuffix("_summary.json")
            pairs.add(_split_harness(raw_slug))
    return sorted(pairs)


_TOK_PER_WORD = 1.3  # rough whitespace-word → BPE-token ratio

# (input_$/1M_tok, output_$/1M_tok) — text only
# slug → (price_in, price_out).  Slug is model_id with "/" replaced by "_".
# Source: https://ai.google.dev/gemini-api/docs/gemini-3
_PRICING: Dict[str, tuple] = {
    "gemini_gemini-3.1-pro-preview":        (2.00, 12.00),
    "gemini_gemini-3-pro-preview":          (2.00, 12.00),
    "gemini_gemini-3-flash-preview":        (0.50,  3.00),
    "gemini_gemini-3.1-flash-lite-preview": (0.25,  1.50),
    "gemini_gemini-2.5-pro":                (1.25, 10.00),
    "gemini_gemini-2.5-flash":              (0.15,  0.60),
    "gemini_gemini-2.0-flash":              (0.10,  0.40),
}

# Task generation always uses 3.1 Pro
_TASK_GEN_MODEL = "gemini_gemini-3.1-pro-preview"
_TASK_GEN_AVG_INPUT_WORDS = 800
_TASK_GEN_AVG_OUTPUT_WORDS = 1200


def _estimate_cost(input_tokens: int, output_tokens: int,
                   model_slug: str) -> float:
    """Return estimated USD cost given token counts and model slug."""
    pi, po = _PRICING.get(model_slug, (1.0, 5.0))
    return (input_tokens * pi + output_tokens * po) / 1e6


def _msg_words(msg: Dict[str, Any]) -> int:
    """Total word count in a single chat message (content + tool-call args)."""
    content = msg.get("content")
    wc = len(content.split()) if content else 0
    for tc in (msg.get("tool_calls") or []):
        args = tc.get("function", {}).get("arguments", "")
        if args:
            wc += len(args.split())
    return wc


def _count_words_in_messages(
    messages: List[Dict[str, Any]],
) -> tuple:
    """Return (input_words, output_words) across all messages."""
    inp, out = 0, 0
    for msg in messages:
        w = _msg_words(msg)
        if msg.get("role") == "assistant":
            out += w
        else:
            inp += w
    return inp, out


def _peak_context_words(
    messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Decompose a run's messages into turn-level word counts.

    Returns a dict with:
      - sys_user_w:     words in leading system + first user message(s)
      - turn_pair_words: per-turn words of (assistant_i + tool_i) for i<T
                         (the new content added to the history after each
                         assistant turn; tool/user content that follows is
                         attributed to the preceding assistant's turn-pair)
      - last_asst_w:    words in the final assistant message
      - prompt_sum_w:   total "input" words summed over all T per-turn prompts
                        (each prompt = system+user + all prior turn-pairs)
      - completion_sum_w: total assistant words across all T turns
      - num_turns:      T (= number of assistant messages)
    Returns None if the run has no assistant message.
    """
    i = 0
    sys_user_w = 0
    while i < len(messages) and messages[i].get("role") in ("system", "user"):
        sys_user_w += _msg_words(messages[i])
        i += 1

    turn_pair_words: List[int] = []
    current_asst_w = 0
    current_extra_w = 0
    saw_asst = False
    asst_word_list: List[int] = []

    for m in messages[i:]:
        role = m.get("role")
        w = _msg_words(m)
        if role == "assistant":
            if saw_asst:
                turn_pair_words.append(current_asst_w + current_extra_w)
                current_extra_w = 0
            current_asst_w = w
            asst_word_list.append(w)
            saw_asst = True
        else:
            if saw_asst:
                current_extra_w += w
    if saw_asst:
        turn_pair_words.append(current_asst_w + current_extra_w)

    T = len(asst_word_list)
    if T == 0:
        return None

    prompt_sum_w = T * sys_user_w
    for j in range(T - 1):
        prompt_sum_w += (T - 1 - j) * turn_pair_words[j]
    completion_sum_w = sum(asst_word_list)

    return {
        "sys_user_w": sys_user_w,
        "turn_pair_words": turn_pair_words,
        "last_asst_w": asst_word_list[-1],
        "prompt_sum_w": prompt_sum_w,
        "completion_sum_w": completion_sum_w,
        "num_turns": T,
    }


def _load_summary(summary_path: Path, record: Dict[str, Any]) -> None:
    """Populate *record* with metrics from a model summary file.

    If the summary contains actual ``usage`` data (prompt_tokens,
    completion_tokens) captured from the API, those are used directly.
    Otherwise we fall back to word-count estimation.
    """
    with open(summary_path) as f:
        sol = json.load(f)
    record["num_runs"] = sol.get("num_runs", 0)
    record["num_success"] = sol.get("num_success", 0)
    raw_pak = sol.get("pass_at_k", {})
    # Coerce keys to ``int`` once so downstream code doesn't have to fight
    # the JSON-string-vs-int duality.  ``run_n_solutions`` writes ``int``,
    # ``run_n_solutions_vanillux`` also writes ``int``, but ``json.dumps`` /
    # ``json.loads`` of a dict-keyed-by-int converts the keys to strings on
    # disk, so what we read back is always strings. Be defensive.
    pass_at_k_full: Dict[int, float] = {}
    for k, v in raw_pak.items():
        try:
            pass_at_k_full[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    record["pass_at_k_full"] = pass_at_k_full
    record["pass_k_max_avail"] = max(pass_at_k_full) if pass_at_k_full else 0
    record["pass@1"] = pass_at_k_full.get(1)
    # ``harness`` was added to summaries by vanillux_solver in commit ad9d7fe.
    # Bash summaries don't carry the field, so we infer "bash" by default and
    # fall back to filename-based detection if the JSON omits it.
    if "harness" in sol:
        record["harness"] = sol["harness"]
    else:
        _, h_from_name = _split_harness(
            summary_path.name.removesuffix("_summary.json")
        )
        record["harness"] = h_from_name

    turns_per_run = []
    input_words_per_run = []
    output_words_per_run = []
    peak_info_per_run: List[Optional[Dict[str, Any]]] = []
    for r in sol.get("results", []):
        msgs = r.get("messages", [])
        n_turns = sum(1 for m in msgs if m.get("role") == "tool")
        turns_per_run.append(n_turns)
        iw, ow = _count_words_in_messages(msgs)
        input_words_per_run.append(iw)
        output_words_per_run.append(ow)
        peak_info_per_run.append(_peak_context_words(msgs))

    n = len(turns_per_run)
    total_in_w = sum(input_words_per_run)
    total_out_w = sum(output_words_per_run)

    record["avg_turns"] = sum(turns_per_run) / n if n else 0
    record["total_words"] = total_in_w + total_out_w
    record["avg_words"] = record["total_words"] / n if n else 0

    # Prefer actual API usage when available; fall back to word-count estimate.
    top_usage = sol.get("usage")
    has_actual = (
        top_usage
        and (top_usage.get("prompt_tokens", 0) or top_usage.get("completion_tokens", 0))
    )

    if has_actual:
        record["total_input_tokens"] = top_usage.get("prompt_tokens", 0)
        record["total_output_tokens"] = top_usage.get("completion_tokens", 0)
        record["reasoning_tokens"] = top_usage.get("reasoning_tokens", 0)
        record["tokens_source"] = "api"
    else:
        # Try summing per-result usage (newer format may have per-result but no top-level)
        per_result_in = 0
        per_result_out = 0
        found_any = False
        for r in sol.get("results", []):
            ru = r.get("usage")
            if ru and (ru.get("prompt_tokens", 0) or ru.get("completion_tokens", 0)):
                per_result_in += ru.get("prompt_tokens", 0)
                per_result_out += ru.get("completion_tokens", 0)
                found_any = True
        if found_any:
            record["total_input_tokens"] = per_result_in
            record["total_output_tokens"] = per_result_out
            record["reasoning_tokens"] = sum(
                (r.get("usage") or {}).get("reasoning_tokens", 0)
                for r in sol.get("results", [])
            )
            record["tokens_source"] = "api"
        else:
            record["total_input_tokens"] = int(total_in_w * _TOK_PER_WORD)
            record["total_output_tokens"] = int(total_out_w * _TOK_PER_WORD)
            record["reasoning_tokens"] = 0
            record["tokens_source"] = "estimated"

    record["total_tokens_est"] = (
        record["total_input_tokens"] + record["total_output_tokens"]
    )
    record["avg_tokens_est"] = record["total_tokens_est"] // n if n else 0

    # ── Per-run peak-context estimates (final turn input / output, initial
    # input), averaged across runs for this task.  Stored tokens-per-word is
    # calibrated per-run to the recorded API usage when available so that
    # summing per-turn inputs reproduces the recorded per-run prompt_tokens.
    initial_in_sum = 0.0
    peak_in_sum = 0.0
    final_out_sum = 0.0
    n_peak = 0
    for r, info in zip(sol.get("results", []), peak_info_per_run):
        if not info:
            continue
        tpw: Optional[float] = None
        ru = r.get("usage") or {}
        pt = ru.get("prompt_tokens", 0) or 0
        ct = ru.get("completion_tokens", 0) or 0
        denom_w = info["prompt_sum_w"] + info["completion_sum_w"]
        if (pt + ct) > 0 and denom_w > 0:
            tpw = (pt + ct) / denom_w
        elif denom_w > 0:
            tpw = _TOK_PER_WORD
        if tpw is None:
            continue
        T = info["num_turns"]
        sys_user_w = info["sys_user_w"]
        pre_last_w = sys_user_w + sum(info["turn_pair_words"][: T - 1])
        initial_in_sum += sys_user_w * tpw
        peak_in_sum += pre_last_w * tpw
        final_out_sum += info["last_asst_w"] * tpw
        n_peak += 1

    if n_peak:
        record["avg_initial_input_tokens"] = initial_in_sum / n_peak
        record["avg_peak_input_tokens"] = peak_in_sum / n_peak
        record["avg_final_output_tokens"] = final_out_sum / n_peak
    else:
        record["avg_initial_input_tokens"] = 0.0
        record["avg_peak_input_tokens"] = 0.0
        record["avg_final_output_tokens"] = 0.0

    record["has_solutions"] = True


def _summary_basename(model_slug: str, harness: str) -> str:
    """Mirror of ``rl_data.generate_solutions._summary_basename`` operating
    on slugs (model_id with ``/`` already replaced by ``_``)."""
    if harness == "bash":
        return f"{model_slug}_summary.json"
    return f"{model_slug}_{harness}_summary.json"


def load_tasks(
    tasks_dir: Path,
    model_slug: Optional[str] = None,
    harness: str = "bash",
) -> List[Dict[str, Any]]:
    """Scan *tasks_dir* for task directories and load metadata + solution summaries.

    If *model_slug* is given (e.g. ``"gemini_gemini-3-flash-preview"``), only
    that model + harness pair's summary is loaded (file named
    ``<slug>_summary.json`` for bash, ``<slug>_<HARNESS>_summary.json`` for
    others — see ``rl_data.generate_solutions._summary_basename``).
    Otherwise the first ``*_summary.json`` found is used.
    """
    records = []
    for task_path in sorted(tasks_dir.iterdir()):
        if not _is_task_dir(task_path):
            continue
        task_json = task_path / "task.json"
        if not task_json.exists():
            continue

        with open(task_json) as f:
            task_data = json.load(f)

        # Adapter-produced tasks (ET, OT-Agent-v1-RL, ...) leave the native
        # taxonomy fields as "unknown" and get LLM-classified values written
        # under `classified_*` by rl_data.comparison.taxonomy_classifier.
        # Prefer those when present, falling back to native fields so
        # skill-tax (which has native taxonomy) keeps working.
        def _pref(*keys: str, default: str = "unknown") -> str:
            for k in keys:
                v = task_data.get(k)
                if v not in (None, "", "unknown"):
                    return v
            return default

        raw_tc = _pref("classified_task_complexity", "task_complexity", "complexity")
        raw_cc = _pref("classified_command_complexity", "command_complexity")

        # v2 axes (commit ad9d7fe). Defaults reproduce the legacy bucket so
        # pre-v2 task.json files (which simply omit these keys) get the same
        # categorical treatment as a corpus_kind="legacy" v2 task.
        verifier_kind = task_data.get("verifier_kind") or "exact_text"
        fixture_kind = task_data.get("fixture_kind") or "text_only"
        corpus_kind = task_data.get("corpus_kind") or "legacy"
        # task_template_gen sets base_image="intricate" only for v2 tasks
        # (any non-legacy verifier/fixture or intricate complexity); pre-v2
        # tasks leave the field unset, which we surface as "per_domain".
        base_image = task_data.get("base_image") or "per_domain"

        record: Dict[str, Any] = {
            "name": task_data.get("name", task_path.name),
            "domain": _pref("classified_domain", "domain", "category"),
            "skill_type": _pref("classified_skill_type", "skill_type"),
            "primitive_skills": (task_data.get("classified_primitive_skills")
                                 or task_data.get("primitive_skills", [])),
            "task_complexity": _shorten_task_complexity(raw_tc),
            "command_complexity": _shorten_cmd_complexity(raw_cc),
            "scenario": _pref("classified_scenario", "scenario"),
            # v2-axis fields — always populated, default to legacy values.
            "verifier_kind": verifier_kind,
            "fixture_kind": fixture_kind,
            "corpus_kind": corpus_kind,
            "base_image": base_image,
            "dir": str(task_path),
        }

        _NO_SOLUTION = dict(
            num_runs=0, num_success=0, avg_turns=0,
            total_words=0, avg_words=0,
            total_input_tokens=0, total_output_tokens=0,
            reasoning_tokens=0,
            total_tokens_est=0, avg_tokens_est=0,
            avg_initial_input_tokens=0.0,
            avg_peak_input_tokens=0.0,
            avg_final_output_tokens=0.0,
            tokens_source="none",
            has_solutions=False,
            harness="none",
            pass_at_k_full={},
            pass_k_max_avail=0,
            **{"pass@1": None},
        )

        solutions_dir = task_path / "solutions"
        if model_slug:
            summary_file = solutions_dir / _summary_basename(model_slug, harness)
            if summary_file.exists():
                _load_summary(summary_file, record)
                record.setdefault("harness", harness)
            else:
                record.update(_NO_SOLUTION)
        else:
            summary_files = (
                list(solutions_dir.glob("*_summary.json"))
                if solutions_dir.exists()
                else []
            )
            summary_files = [f for f in summary_files if f.name != "summary.json"]
            if summary_files:
                _load_summary(summary_files[0], record)
            else:
                record.update(_NO_SOLUTION)

        records.append(record)
    return records


def _fmt_count(n: int) -> str:
    """Human-friendly large number: 1234 → '1,234', 1234567 → '1.23M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def _pass_at_k(record: Dict[str, Any], k: int) -> Optional[float]:
    """Look up pass@k for a record (returns None if the task has no value at
    this specific k — e.g. it was sampled at NUM_SOLUTIONS=4 and we asked
    for k=8)."""
    if not record.get("has_solutions"):
        return None
    return record.get("pass_at_k_full", {}).get(k)


def print_summary(
    records: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    model_slug: Optional[str] = None,
    max_rows: int = 50,
    harness: str = "bash",
    pass_k_target: int = 8,
) -> None:
    """Print model stats banner, optionally per-task rows, and aggregate table.

    *pass_k_target* is the "ceiling" k whose mean is shown alongside pass@1
    in the banner / per-axis tables. Resolved by ``_resolve_pass_k`` (corpus
    common max k, or user override). Tasks without a value at this k are
    excluded from that specific cell — callers should pick a k that all
    solved tasks have to keep aggregates apples-to-apples.
    """
    solved = [r for r in records if r["has_solutions"]]

    # ── Model-level stats banner ──────────────────────────────────────
    total_runs = sum(r["num_runs"] for r in records)
    total_words = sum(r["total_words"] for r in records)
    total_in_tok = sum(r["total_input_tokens"] for r in records)
    total_out_tok = sum(r["total_output_tokens"] for r in records)
    total_tokens = total_in_tok + total_out_tok
    avg_words = int(total_words / total_runs) if total_runs else 0
    avg_tokens = int(total_tokens / total_runs) if total_runs else 0

    title = model_name or "all models"
    if harness and harness != "bash":
        title = f"{title}  [harness: {harness}]"
    print(f"\n{'═'*80}")
    print(f"  {title}")
    print(f"{'═'*80}")
    print(
        f"  Tasks: {len(records)}  "
        f"│  With solutions: {len(solved)}  "
        f"│  Total runs: {total_runs}"
    )

    # Surface the per-corpus k mix so it's obvious when a corpus is ragged
    # (e.g. some tasks sampled at NUM_SOLUTIONS=4, others at 8).
    if solved:
        ks_max = Counter(r["pass_k_max_avail"] for r in solved)
        if len(ks_max) > 1:
            mix = ", ".join(
                f"k≤{k}: {n}" for k, n in sorted(ks_max.items())
            )
            print(
                f"  Pass@k ceiling: k={pass_k_target} "
                f"(per-task max-k mix: {mix})"
            )
        elif pass_k_target:
            (only_k,) = list(ks_max)
            print(f"  Pass@k ceiling: k={pass_k_target} "
                  f"(every solved task has k=1..{only_k})")

    pk_label = f"p@{pass_k_target}" if pass_k_target else "p@-"
    if solved:
        avg_p1 = (
            sum(r["pass@1"] for r in solved if r["pass@1"] is not None)
            / len(solved)
        )
        pk_vals = [_pass_at_k(r, pass_k_target) for r in solved]
        pk_vals = [v for v in pk_vals if v is not None]
        avg_pk = sum(pk_vals) / len(pk_vals) if pk_vals else 0.0
        avg_turns = sum(r["avg_turns"] for r in solved) / len(solved)
        pk_n_label = (
            f"  (n={len(pk_vals)}/{len(solved)})"
            if len(pk_vals) != len(solved) else ""
        )
        print(
            f"  Mean p@1: {avg_p1:.2f}  "
            f"│  Mean {pk_label}: {avg_pk:.2f}{pk_n_label}  "
            f"│  Avg turns: {avg_turns:.1f}"
        )
    print(
        f"  Total words: {_fmt_count(total_words)}  "
        f"│  Avg words/run: {_fmt_count(avg_words)}  "
        f"│  Avg tokens/run: {_fmt_count(avg_tokens)}"
    )
    # Determine token source label
    sources = {r.get("tokens_source", "none") for r in records if r["has_solutions"]}
    if sources == {"api"}:
        tok_label = "actual"
    elif "api" in sources:
        tok_label = "actual+est"
    else:
        tok_label = "est"

    total_reasoning = sum(r.get("reasoning_tokens", 0) for r in records)
    reasoning_info = ""
    if total_reasoning:
        reasoning_info = f"  (incl. {_fmt_count(total_reasoning)} reasoning)"

    print(
        f"  Input tokens: {_fmt_count(total_in_tok)}  "
        f"│  Output tokens: {_fmt_count(total_out_tok)}  "
        f"│  Total tokens: {_fmt_count(total_tokens)}  ({tok_label})"
    )
    if reasoning_info:
        print(f"  {reasoning_info}")

    # Cost estimation
    slug = model_slug
    if slug and slug in _PRICING:
        pi, po = _PRICING[slug]
        sol_cost = _estimate_cost(total_in_tok, total_out_tok, slug)

        # Task generation cost (always 3.1 Pro)
        n_tasks = len(records)
        tg_in = int(n_tasks * _TASK_GEN_AVG_INPUT_WORDS * _TOK_PER_WORD)
        tg_out = int(n_tasks * _TASK_GEN_AVG_OUTPUT_WORDS * _TOK_PER_WORD)
        tg_cost = _estimate_cost(tg_in, tg_out, _TASK_GEN_MODEL)

        cost_per_run = sol_cost / total_runs if total_runs else 0
        print(
            f"  Solution cost (est): ${sol_cost:,.2f}  "
            f"(${pi:.2f}/${po:.2f} per 1M tok)  "
            f"│  $/run: ${cost_per_run:,.4f}"
        )
        print(
            f"  Task gen cost (est): ${tg_cost:,.2f}  "
            f"(3.1-pro, {n_tasks} tasks)  "
            f"│  Total pipeline: ${tg_cost + sol_cost:,.2f}"
        )
    elif slug:
        print(f"  Cost: pricing not available for {slug}")

    print(f"{'─'*80}")

    # ── Per-task rows (skip if too many) ──────────────────────────────
    show_rows = max_rows == 0 or len(records) <= max_rows
    if show_rows:
        pk_col = f"p@{pass_k_target}" if pass_k_target else "p@-"
        header = (
            f"{'Task':<30} {'Domain':<24} {'Skill Type':<20} "
            f"{'Task Cplx':<12} {'Cmd Cplx':<20} "
            f"{'Runs':>5} {'Pass':>5} "
            f"{'p@1':>6} {pk_col:>6} {'Turns':>6}"
        )
        print(header)
        print("-" * len(header))
        for r in records:
            p1 = f"{r['pass@1']:.2f}" if r["pass@1"] is not None else "-"
            pk_val = _pass_at_k(r, pass_k_target)
            pk = f"{pk_val:.2f}" if pk_val is not None else "-"
            turns = (
                f"{r['avg_turns']:>6.1f}" if r["has_solutions"] else f"{'-':>6}"
            )
            print(
                f"{r['name']:<30} {r['domain']:<24} {r['skill_type']:<20} "
                f"{r['task_complexity']:<12} {r['command_complexity']:<20} "
                f"{r['num_runs']:>5} {r['num_success']:>5} "
                f"{p1:>6} {pk:>6} {turns}"
            )
        print()
    else:
        print(
            f"  ({len(records)} tasks — per-task rows hidden; "
            f"use --max-rows 0 to show all)\n"
        )

    # ── Aggregate breakdown ───────────────────────────────────────────
    if not solved:
        return
    _print_aggregate(solved, "domain", "Domain", model_slug=slug,
                     pass_k_target=pass_k_target)
    _print_aggregate(solved, "task_complexity", "Task Complexity",
                     key_order=_TASK_COMPLEXITY_ORDER, model_slug=slug,
                     pass_k_target=pass_k_target)
    _print_aggregate(solved, "command_complexity", "Cmd Complexity",
                     key_order=_CMD_COMPLEXITY_ORDER, model_slug=slug,
                     pass_k_target=pass_k_target)

    # ── v2-axis breakdowns (only print when the axis isn't degenerate) ─
    # Suppresses pure noise on legacy-only corpora where every task has the
    # same default value, while still surfacing the breakdown the moment a
    # corpus mixes legacy + v2 tasks.
    def _has_variation(field: str) -> bool:
        return len({r[field] for r in solved}) > 1

    if _has_variation("verifier_kind"):
        _print_aggregate(solved, "verifier_kind", "Verifier Kind",
                         key_order=_VERIFIER_KIND_ORDER, model_slug=slug,
                         pass_k_target=pass_k_target)
    if _has_variation("fixture_kind"):
        _print_aggregate(solved, "fixture_kind", "Fixture Kind",
                         key_order=_FIXTURE_KIND_ORDER, model_slug=slug,
                         pass_k_target=pass_k_target)
    if _has_variation("corpus_kind"):
        _print_aggregate(solved, "corpus_kind", "Corpus Kind",
                         key_order=_CORPUS_KIND_ORDER, model_slug=slug,
                         pass_k_target=pass_k_target)
    if _has_variation("base_image"):
        _print_aggregate(solved, "base_image", "Base Image",
                         key_order=_BASE_IMAGE_ORDER, model_slug=slug,
                         pass_k_target=pass_k_target)


def _print_aggregate(
    solved: List[Dict[str, Any]],
    field: str,
    label: str,
    key_order: Optional[List[str]] = None,
    model_slug: Optional[str] = None,
    pass_k_target: int = 8,
) -> None:
    """Print a small aggregate table grouped by *field*.

    The ``p@K`` column header / lookup uses *pass_k_target* (resolved by
    ``_resolve_pass_k`` to keep cross-task means apples-to-apples). Tasks
    in a bucket that don't have a value at this k are excluded from that
    bucket's pass@K average only — pass@1 still uses every task.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in solved:
        buckets[r[field]].append(r)

    keys = key_order if key_order else sorted(buckets.keys())
    has_cost = model_slug and model_slug in _PRICING
    pk_col = f"p@{pass_k_target}" if pass_k_target else "p@-"

    hdr = (
        f"  {'':2}{label:<24} {'n':>4} {'p@1':>6} {pk_col:>6} "
        f"{'Turns':>6} {'Tokens':>10}"
    )
    if has_cost:
        hdr += f" {'Cost':>10}"
    print(hdr)
    print(f"  {'':2}{'-'*(len(hdr)-4)}")
    for k in keys:
        recs = buckets.get(k, [])
        if not recs:
            line = (f"  {'':2}{k:<24} {'0':>4} {'-':>6} {'-':>6} "
                    f"{'-':>6} {'-':>10}")
            if has_cost:
                line += f" {'-':>10}"
            print(line)
            continue
        n = len(recs)
        mp1 = sum(r["pass@1"] for r in recs if r["pass@1"] is not None) / n
        pk_vals = [_pass_at_k(r, pass_k_target) for r in recs]
        pk_vals = [v for v in pk_vals if v is not None]
        mpk_str = f"{(sum(pk_vals) / len(pk_vals)):>6.2f}" if pk_vals else f"{'-':>6}"
        mt = sum(r["avg_turns"] for r in recs) / n
        ti = sum(r["total_input_tokens"] for r in recs)
        to = sum(r["total_output_tokens"] for r in recs)
        tt = ti + to
        line = (
            f"  {'':2}{k:<24} {n:>4} {mp1:>6.2f} {mpk_str} "
            f"{mt:>6.1f} {_fmt_count(tt):>10}"
        )
        if has_cost:
            c = _estimate_cost(ti, to, model_slug)
            line += f" {'${:,.2f}'.format(c):>10}"
        print(line)
    print()


def plot_distributions(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate pie charts for all metadata axes (legacy + v2)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Legacy axes — always plotted for backwards-compatibility.
    axes = [
        ("domain", "Domain Distribution", "dist_domain.png", None),
        ("skill_type", "Skill Type Distribution", "dist_skill_type.png", None),
        ("task_complexity", "Task Complexity Distribution",
         "dist_task_complexity.png", _TASK_COMPLEXITY_ORDER),
        (
            "command_complexity",
            "Command Complexity Distribution",
            "dist_command_complexity.png",
            _CMD_COMPLEXITY_ORDER,
        ),
        ("scenario", "Scenario Distribution", "dist_scenario.png", None),
    ]

    # v2 axes (commit ad9d7fe). Always plotted — for a legacy-only corpus
    # the verifier/fixture/corpus pies will simply collapse to a single
    # "exact_text" / "text_only" / "legacy" / "per_domain" wedge, which is
    # itself useful as a sanity check.
    axes.extend([
        ("verifier_kind", "Verifier Kind Distribution",
         "dist_verifier_kind.png", _VERIFIER_KIND_ORDER),
        ("fixture_kind", "Fixture Kind Distribution",
         "dist_fixture_kind.png", _FIXTURE_KIND_ORDER),
        ("corpus_kind", "Corpus Kind Distribution",
         "dist_corpus_kind.png", _CORPUS_KIND_ORDER),
        ("base_image", "Base Image Distribution",
         "dist_base_image.png", _BASE_IMAGE_ORDER),
    ])

    for field, title, fname, order in axes:
        counts = Counter(r[field] for r in records)
        if order is not None:
            ordered_keys = [k for k in order if counts.get(k)]
            extra_keys = sorted(k for k in counts if k not in order)
            labels = ordered_keys + extra_keys
        else:
            labels = list(counts.keys())
        sizes = [counts[k] for k in labels]
        if not sizes:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        wedges, _texts, _autotexts = ax.pie(
            sizes,
            labels=None,
            autopct="%1.0f%%",
            startangle=90,
            pctdistance=0.85,
            textprops={"fontsize": 9},
        )
        ax.legend(
            wedges,
            [f"{lb} ({ct})" for lb, ct in zip(labels, sizes)],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / fname}")


def _bar_chart(
    records: List[Dict[str, Any]],
    field: str,
    metric: str,
    ylabel: str,
    title: str,
    fname: str,
    out_dir: Path,
    color: str = "steelblue",
    expected_keys: Optional[List[str]] = None,
) -> None:
    """Helper: grouped bar chart of *metric* averaged by *field*.

    If *expected_keys* is given, all listed categories are shown (in that
    order) even when no data exists for some of them.
    """
    buckets: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        val = r.get(metric)
        if val is not None:
            buckets[r[field]].append(val)
    if not buckets and not expected_keys:
        return

    if expected_keys:
        keys = expected_keys
    else:
        keys = sorted(buckets.keys())

    means = [
        (sum(buckets[k]) / len(buckets[k])) if buckets.get(k) else 0
        for k in keys
    ]
    counts = [len(buckets.get(k, [])) for k in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 5))
    bars = ax.bar(range(len(keys)), means, color=color)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if metric.startswith("pass"):
        ax.set_ylim(0, 1.05)
    for bar, val, n in zip(bars, means, counts):
        if n == 0:
            label = "n=0"
        elif metric.startswith("pass"):
            label = f"{val:.2f}\n(n={n})"
        else:
            label = f"{val:.1f}\n(n={n})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / fname}")


def plot_quality(
    records: List[Dict[str, Any]],
    out_dir: Path,
    model_name: Optional[str] = None,
    model_slug: Optional[str] = None,
    harness: str = "bash",
    pass_k_target: int = 8,
) -> None:
    """Generate quality analysis plots (bar charts + pass@k curve).

    *pass_k_target* is the "ceiling" k for the second-row bar charts (Pass@K
    by Domain, by Task Complexity, ...). Resolved per-corpus so a smoke run
    sampled at NUM_SOLUTIONS=4 produces honest pass@4 plots, not pass@8
    plots full of "-" cells. Tasks without a value at this k are excluded
    from that specific aggregate, with a per-bar ``n=`` label so it's
    visible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    solved = [r for r in records if r["has_solutions"] and r["pass@1"] is not None]
    if not solved:
        print("  No solution data available for quality plots.")
        return

    # Project the dynamic pass@K value onto a stable record key the
    # _bar_chart helper can read by name. We materialise it here (instead of
    # threading a callable into _bar_chart) because some records may not
    # have a value at K — _bar_chart already skips ``None`` values.
    pk_field = f"pass@k_target"
    for r in solved:
        r[pk_field] = _pass_at_k(r, pass_k_target)

    tag_pieces = []
    if model_name:
        tag_pieces.append(model_name)
    if harness and harness != "bash":
        tag_pieces.append(harness)
    tag = f" [{' / '.join(tag_pieces)}]" if tag_pieces else ""
    all_domains = sorted({r["domain"] for r in records})
    pk_label = f"Pass@{pass_k_target}" if pass_k_target else "Pass@-"
    pk_fname = f"pass{pass_k_target}" if pass_k_target else "passK"

    # -- pass@1 charts --
    _bar_chart(
        solved, "domain", "pass@1", "Mean pass@1",
        f"Pass@1 by Domain{tag}", "quality_pass1_by_domain.png",
        out_dir, color="steelblue", expected_keys=all_domains,
    )
    _bar_chart(
        solved, "task_complexity", "pass@1", "Mean pass@1",
        f"Pass@1 by Task Complexity{tag}", "quality_pass1_by_task_complexity.png",
        out_dir, color="darkorange", expected_keys=_TASK_COMPLEXITY_ORDER,
    )
    _bar_chart(
        solved, "command_complexity", "pass@1", "Mean pass@1",
        f"Pass@1 by Command Complexity{tag}", "quality_pass1_by_command_complexity.png",
        out_dir, color="mediumpurple", expected_keys=_CMD_COMPLEXITY_ORDER,
    )

    # -- pass@K (pass-at-any) charts — K is resolved per-corpus --
    _bar_chart(
        solved, "domain", pk_field, f"Mean {pk_label.lower()}",
        f"{pk_label} by Domain{tag}",
        f"quality_{pk_fname}_by_domain.png",
        out_dir, color="royalblue", expected_keys=all_domains,
    )
    _bar_chart(
        solved, "task_complexity", pk_field, f"Mean {pk_label.lower()}",
        f"{pk_label} by Task Complexity{tag}",
        f"quality_{pk_fname}_by_task_complexity.png",
        out_dir, color="coral", expected_keys=_TASK_COMPLEXITY_ORDER,
    )
    _bar_chart(
        solved, "command_complexity", pk_field, f"Mean {pk_label.lower()}",
        f"{pk_label} by Command Complexity{tag}",
        f"quality_{pk_fname}_by_command_complexity.png",
        out_dir, color="orchid", expected_keys=_CMD_COMPLEXITY_ORDER,
    )

    # -- turns charts --
    _bar_chart(
        solved, "task_complexity", "avg_turns", "Avg Turns",
        f"Average Turns by Task Complexity{tag}",
        "quality_turns_by_task_complexity.png",
        out_dir, color="seagreen", expected_keys=_TASK_COMPLEXITY_ORDER,
    )
    _bar_chart(
        solved, "domain", "avg_turns", "Avg Turns",
        f"Average Turns by Domain{tag}", "quality_turns_by_domain.png",
        out_dir, color="teal", expected_keys=all_domains,
    )

    # -- v2-axis quality plots (only render when the axis varies in the
    #    solved set; otherwise a single bar at 100 % carries no info). ----
    def _seen_keys(field: str, order: List[str]) -> Optional[List[str]]:
        seen = {r[field] for r in solved}
        if len(seen) <= 1:
            return None
        return [k for k in order if k in seen] + sorted(seen - set(order))

    for field, order, color_p1, color_p8, color_t in [
        ("verifier_kind", _VERIFIER_KIND_ORDER,
         "steelblue", "royalblue", "teal"),
        ("fixture_kind", _FIXTURE_KIND_ORDER,
         "darkorange", "coral", "seagreen"),
        ("corpus_kind", _CORPUS_KIND_ORDER,
         "mediumpurple", "orchid", "darkslateblue"),
        ("base_image", _BASE_IMAGE_ORDER,
         "saddlebrown", "peru", "olive"),
    ]:
        keys = _seen_keys(field, order)
        if keys is None:
            continue
        pretty = field.replace("_", " ").title()
        _bar_chart(
            solved, field, "pass@1", "Mean pass@1",
            f"Pass@1 by {pretty}{tag}", f"quality_pass1_by_{field}.png",
            out_dir, color=color_p1, expected_keys=keys,
        )
        _bar_chart(
            solved, field, pk_field, f"Mean {pk_label.lower()}",
            f"{pk_label} by {pretty}{tag}",
            f"quality_{pk_fname}_by_{field}.png",
            out_dir, color=color_p8, expected_keys=keys,
        )
        _bar_chart(
            solved, field, "avg_turns", "Avg Turns",
            f"Average Turns by {pretty}{tag}", f"quality_turns_by_{field}.png",
            out_dir, color=color_t, expected_keys=keys,
        )

    # --- Pass@k curve (averaged across tasks) ---
    # Build directly from records (which already cache pass_at_k_full) so we
    # don't re-read JSON. Plot every k present, but draw a vertical guide
    # line at the corpus-wide common ceiling K to make ragged k corpora
    # visually obvious. Each point is annotated with n = #tasks contributing
    # at that k.
    all_pass_at_k: Dict[int, List[float]] = defaultdict(list)
    for r in solved:
        for k, v in (r.get("pass_at_k_full") or {}).items():
            all_pass_at_k[k].append(v)

    if all_pass_at_k:
        ks = sorted(all_pass_at_k.keys())
        means = [sum(all_pass_at_k[k]) / len(all_pass_at_k[k]) for k in ks]
        ns = [len(all_pass_at_k[k]) for k in ks]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ks, means, "o-", color="crimson", linewidth=2, markersize=5)
        for kk, m, nn in zip(ks, means, ns):
            ax.annotate(
                f"n={nn}", xy=(kk, m), xytext=(0, 6),
                textcoords="offset points", fontsize=7,
                ha="center", color="dimgray",
            )
        ax.set_xlabel("k")
        ax.set_ylabel("Mean pass@k")
        ax.set_title(
            f"Pass@k Curve (averaged across tasks){tag}", fontweight="bold",
        )
        ax.set_ylim(0, 1.10)
        ax.grid(True, alpha=0.3)
        # Draw a guide line at the corpus-wide common ceiling K — beyond
        # this k the curve is averaged over a strict subset of tasks (tasks
        # that were sampled at a higher NUM_SOLUTIONS), so the right tail
        # is not directly comparable to the left.
        if pass_k_target and pass_k_target in all_pass_at_k:
            ax.axvline(
                pass_k_target, color="gray", linestyle="--", alpha=0.5,
                label=f"common ceiling k={pass_k_target}",
            )
            ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "quality_pass_at_k.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / 'quality_pass_at_k.png'}")


def _analyze_model(
    tasks_dir: Path,
    plots_base: Path,
    model_slug: str,
    max_rows: int = 50,
    harness: str = "bash",
    pass_k_override: Optional[int] = None,
) -> None:
    """Run the full per-model analysis (table + quality plots).

    *harness* selects between bash (legacy filename ``<slug>_summary.json``)
    and any v2 harness (suffixed filename ``<slug>_<HARNESS>_summary.json``,
    e.g. ``vanillux``).

    *pass_k_override*: when None (default), the "ceiling k" for aggregates
    is auto-detected as the largest k that every solved task has data for
    (i.e. ``min(NUM_SOLUTIONS over solved tasks)``). When set, that k is
    used directly — useful for comparing two corpora at a fixed k.
    """
    display = model_slug.replace("_", "/", 1)

    records = load_tasks(tasks_dir, model_slug=model_slug, harness=harness)

    # Resolve the per-corpus pass@K ceiling once. If no solved tasks exist
    # we fall back to k=8 (legacy default) so the column header still says
    # something sensible — every cell will be "-" anyway.
    pk_target = _resolve_pass_k(records, override=pass_k_override) or 8

    # Keep bash output dirs at the legacy location; isolate non-bash harness
    # plots so a side-by-side bash+vanillux run on the same model produces
    # two cleanly-separated plot subtrees.
    sub = model_slug if harness == "bash" else f"{model_slug}__{harness}"
    model_dir = plots_base / sub
    print_summary(records, model_name=display, model_slug=model_slug,
                  max_rows=max_rows, harness=harness,
                  pass_k_target=pk_target)

    label = display if harness == "bash" else f"{display} [{harness}]"
    print(f"Generating quality plots for {label}...")
    plot_quality(records, model_dir, model_name=display,
                 model_slug=model_slug, harness=harness,
                 pass_k_target=pk_target)

    print(f"Done. Model plots saved to {model_dir}/")


def main():
    ap = argparse.ArgumentParser(
        description="Analyze generated RL tasks and solutions."
    )
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Directory containing task_* subdirectories",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Where to save plots (default: <tasks-dir>/analysis)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model to analyze, e.g. 'gemini/gemini-3-flash-preview'. "
            "Omit to auto-discover and analyze all models."
        ),
    )
    ap.add_argument(
        "--harness",
        type=str,
        default=None,
        choices=("bash", "vanillux"),
        help=(
            "Solver harness to analyze. ``bash`` reads the legacy "
            "``<MODEL>_summary.json``; ``vanillux`` reads "
            "``<MODEL>_vanillux_summary.json`` (commit ad9d7fe). "
            "Omit to auto-discover all (model, harness) pairs."
        ),
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help=(
            "Max per-task rows to print (default 50). "
            "Set to 0 to show all rows regardless of count."
        ),
    )
    ap.add_argument(
        "--pass-at-k",
        type=int,
        default=None,
        help=(
            "Override the 'ceiling' k used in pass@K aggregates and bar "
            "charts. Default: auto-detect the largest k that every solved "
            "task has data for (= min of NUM_SOLUTIONS across solved tasks). "
            "This makes the smoke-vs-full comparison honest — a 50-task "
            "smoke sampled at NUM_SOLUTIONS=4 produces pass@4 plots, not "
            "pass@8 plots full of '-'."
        ),
    )
    args = ap.parse_args()

    tasks_dir = args.tasks_dir
    plots_dir = args.plots_dir or (tasks_dir / "analysis")

    print(f"Scanning {tasks_dir}...")

    # Distribution plots use all tasks (model-independent)
    all_records = load_tasks(tasks_dir)
    if not all_records:
        print("No tasks found.")
        return

    print("Generating distribution plots...")
    plot_distributions(all_records, plots_dir)

    # Determine which (model, harness) pair(s) to analyze
    if args.model:
        slug = args.model.replace("/", "_")
        if args.harness:
            pairs = [(slug, args.harness)]
        else:
            # If a model is given but the harness isn't, prefer bash for
            # backwards compatibility but also analyze any non-bash harness
            # summaries that happen to exist on disk.
            pairs = [(slug, "bash")]
            for h in _KNOWN_HARNESSES:
                # Cheap probe — does ANY task have a summary for this harness?
                probe = list(tasks_dir.glob(
                    f"*/solutions/{slug}_{h}_summary.json"
                ))
                if probe:
                    pairs.append((slug, h))
    else:
        pairs = discover_models(tasks_dir)
        if args.harness:
            pairs = [(s, h) for (s, h) in pairs if h == args.harness]
        if not pairs:
            print("No model summaries found — nothing to analyze.")
            return
        pretty = ", ".join(
            s if h == "bash" else f"{s} [{h}]" for s, h in pairs
        )
        print(f"Discovered {len(pairs)} (model, harness) pair(s): {pretty}")

    for slug, harness in pairs:
        _analyze_model(
            tasks_dir, plots_dir, slug,
            max_rows=args.max_rows, harness=harness,
            pass_k_override=args.pass_at_k,
        )

    print(f"\nDone. All plots saved under {plots_dir}/")


if __name__ == "__main__":
    main()
