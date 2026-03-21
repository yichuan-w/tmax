"""Build, test, and run solutions for generated tasks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TextIO

from tqdm import tqdm

from rl_data import DEFAULT_MODEL
from rl_data.generate_tasks import _safe_write_text
from rl_data.generator.sample_solutions import run_n_solutions


@dataclass
class SolutionConfig:
    """Configuration for running solutions on tasks."""

    tasks_dir: str
    num_solutions: int = 128
    max_actions: int = 16
    model: str = DEFAULT_MODEL
    solution_temperature: float = 0.7
    verbose: bool = False
    num_tasks: int = 1
    start_at: int = 0
    num_pool_workers: int = 128
    workers: int = 1
    force_build: bool = False
    max_tokens: int = 65536
    filter_solved: bool = False
    use_parquet: bool = False
    command_timeout: float = 30.0
    #: If False, skip task dirs that already have a `*_summary.json` (default).
    #: Set True (--force-rerun) to regenerate solutions and overwrite summaries.
    force_rerun: bool = False
    shell_init_timeout: float = 120.0
    shell_init_attempts: int = 3
    log_commands: bool = False
    #: Relative to each task dir if not absolute; default when log_commands: solutions/debug_commands
    command_log_dir: Optional[str] = None
    #: If set, copy everything printed to stdout/stderr (terminal) into this file (append).
    terminal_log: Optional[str] = None


class _TeeTextStream:
    """Write to the real terminal stream and to a log file (for debugging full run output)."""

    def __init__(self, primary: TextIO, log_file: TextIO) -> None:
        self._primary = primary
        self._log = log_file

    def write(self, data: str) -> int:
        n = self._primary.write(data)
        self._primary.flush()
        self._log.write(data)
        self._log.flush()
        return int(n) if isinstance(n, int) else len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._primary.fileno()


def build_and_test(
    sif_path: Path, def_path: Path, test_py: str, run_initial_tests: bool = True
) -> tuple[bool, str]:
    """Build container and optionally run initial tests."""
    with open(def_path, "r") as f:
        def_text = f.read()

    if "chmod 755 /home/user" not in def_text:
        section_headers = [line for line in def_text.split("\n") if line.strip().startswith("%")]
        post_idx = [i for i, line in enumerate(section_headers) if "post" in line.lower()]
        if post_idx:
            idx = post_idx[0]
            if idx + 1 < len(section_headers):
                next_header = section_headers[idx + 1]
                def_text = def_text.replace(
                    next_header, "    chmod 755 /home/user\n" + next_header
                )
            else:
                def_text = def_text.rstrip() + "\n    chmod 755 /home/user\n"
            with open(def_path, "w") as f:
                f.write(def_text)

    build_rc = subprocess.run(
        ["apptainer", "build", str(sif_path), str(def_path)],
        capture_output=True,
        text=True,
        timeout=180,
    ).returncode
    if build_rc:
        print(f"Apptainer build failed: {build_rc}")
        return False, "Apptainer build failed"

    if not run_initial_tests:
        return True, ""

    test_file = sif_path.parent / "test_initial_state.py"
    test_file.write_text(test_py)

    proc = subprocess.run(
        [
            "apptainer", "exec",
            "--fakeroot", "--userns",
            "--writable-tmpfs", "--cleanenv",
            str(sif_path),
            "pytest", "-q", str(test_file.name),
        ],
        capture_output=True,
        text=True,
    )

    return proc.returncode == 0, proc.stdout + proc.stderr


def process_task(task_dir: str, cfg: SolutionConfig):
    """Process a single task: build, test, run solutions, and cleanup."""
    task_dir = Path(task_dir)
    print(f"\nProcessing task: {task_dir.name}")

    sif_path = task_dir / "container.sif"
    def_path = task_dir / "container.def"
    initial_test_path = task_dir / "test_initial_state.py"
    final_test_path = task_dir / "test_final_state.py"
    task_json_path = task_dir / "task.json"
    solutions_dir = task_dir / "solutions"

    print(f"{task_dir} sif_path: {sif_path}")
    pass_at_k = None

    if not sif_path.exists():
        if not def_path.exists():
            print(f"[{task_dir.name}] No def file found, skipping.")
            return "no def"
        print(f"[{task_dir.name}] Building SIF from def...")
        ok, msg = build_and_test(sif_path, def_path, initial_test_path.read_text(), run_initial_tests=False)
        if not ok:
            print(f"[{task_dir.name}] SIF build failed: {msg}")
            return "no sif"

    try:
        print(f"[{task_dir.name}] Running {cfg.num_solutions} solutions...")
        solutions_dir.mkdir(exist_ok=True)

        cmd_log_resolved: Optional[Path] = None
        if cfg.log_commands:
            if cfg.command_log_dir:
                p = Path(cfg.command_log_dir).expanduser()
                cmd_log_resolved = p if p.is_absolute() else (task_dir / p)
            else:
                cmd_log_resolved = solutions_dir / "debug_commands"
            cmd_log_resolved = cmd_log_resolved.resolve()
            print(f"[{task_dir.name}] Command debug logs -> {cmd_log_resolved}")

        summary = run_n_solutions(
            num_solutions=cfg.num_solutions,
            container_sif_path=str(sif_path),
            initial_test_path=str(initial_test_path),
            final_test_path=str(final_test_path),
            def_path=str(def_path),
            task_path=str(task_json_path),
            max_actions=cfg.max_actions,
            model=cfg.model,
            temperature=cfg.solution_temperature,
            max_tokens=cfg.max_tokens,
            save_dir=str(solutions_dir),
            verbose=cfg.verbose,
            num_pool_workers=cfg.num_pool_workers,
            run_initial_tests=False,
            command_timeout=cfg.command_timeout,
            shell_init_timeout=cfg.shell_init_timeout,
            shell_init_attempts=cfg.shell_init_attempts,
            log_commands=cfg.log_commands,
            command_log_dir=str(cmd_log_resolved) if cmd_log_resolved else None,
        )

        model_name = cfg.model.replace("/", "_")
        _safe_write_text(
            task_dir / "solutions" / f"{model_name}_summary.json",
            json.dumps(summary, indent=4),
        )
        pass_at_k = summary.get("pass_at_k", {})

    finally:
        if sif_path.exists():
            print(f"[{task_dir.name}] Not deleting SIF file.")

    return pass_at_k


def parse_args(argv: Optional[List[str]] = None) -> SolutionConfig:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Build, test, and run solutions for generated tasks."
    )
    ap.add_argument("--tasks-dir", type=str, required=True, help="Directory containing generated tasks")
    ap.add_argument("--start-at", type=int, default=0, help="Start at task number")
    ap.add_argument("--num-tasks", type=int, default=200, help="Number of tasks to process")
    ap.add_argument("--num-solutions", type=int, default=16, help="Number of solution attempts per task")
    ap.add_argument("--max-actions", type=int, default=16, help="Max shell actions per solution attempt")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--solution-temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=65536, help="Max tokens for the solution agent")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    ap.add_argument("--num-pool-workers", type=int, default=128, help="Number of pool workers")
    ap.add_argument("--workers", type=int, default=1, help="Number of concurrent tasks to process")
    ap.add_argument("--force-build", action="store_true", help="Force build the SIF file")
    ap.add_argument("--filter-solved", action="store_true", help="Only solve tasks that have been solved already")
    ap.add_argument("--use-parquet", action="store_true", help="Use parquet file for tasks")
    ap.add_argument("--command-timeout", type=float, default=30.0, help="Per-command timeout in seconds inside containers (default: 30)")
    ap.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run solution generation even when a *_summary.json already exists (overwrites on success)",
    )
    ap.add_argument(
        "--shell-init-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for Apptainer interactive shell init marker (raise if many concurrent containers)",
    )
    ap.add_argument(
        "--shell-init-attempts",
        type=int,
        default=3,
        help="Retries if shell init times out (default: 3)",
    )
    ap.add_argument(
        "--log-commands",
        action="store_true",
        help="Write each container bash command and raw output to per-solution log files under the task (see --command-log-dir)",
    )
    ap.add_argument(
        "--command-log-dir",
        type=str,
        default=None,
        help="Directory for command debug logs: absolute path, or relative to each task folder (default: <task>/solutions/debug_commands)",
    )
    ap.add_argument(
        "--terminal-log",
        type=str,
        default=None,
        metavar="PATH",
        help="Append full stdout/stderr of this process (everything on your central terminal) to PATH; still prints to terminal",
    )

    args = ap.parse_args(argv)
    return SolutionConfig(**vars(args))


def _run_generate_solutions(cfg: SolutionConfig) -> None:
    """Core driver (stdout/stderr may be teed by main())."""
    all_entries = list(Path(cfg.tasks_dir).iterdir())
    task_dirs = [
        d
        for d in tqdm(all_entries, desc="Scanning task directories", total=len(all_entries))
        if d.name.startswith("task_")
    ]

    if cfg.filter_solved:
        print(f"Filtering to tasks with existing pass@16 > 0, prefilter: {len(task_dirs)}")

        def _pass16_gt_zero(task_dir: str) -> bool:
            task_dir = Path(task_dir)
            try:
                model_name = cfg.model.replace("/", "_")
                model_summary_path = task_dir / "solutions" / f"{model_name}_summary.json"
                if model_summary_path.exists():
                    return False
                # Check any existing summary
                summaries = list((task_dir / "solutions").glob("*_summary.json"))
                if not summaries:
                    return False
                with open(summaries[0], "r") as f:
                    data = json.load(f)
                pass_at_k = data.get("pass_at_k", {})
                value = pass_at_k.get("16") or pass_at_k.get(16)
                if value is None:
                    return False
                return float(value) > 0.0
            except Exception:
                return False

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=len(task_dirs)) as executor:
            futures = {executor.submit(_pass16_gt_zero, d): i for i, d in enumerate(task_dirs)}
            mask = [False] * len(task_dirs)
            with tqdm(total=len(task_dirs), desc="Reading summaries") as pbar:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        mask[idx] = fut.result()
                    except Exception:
                        mask[idx] = False
                    finally:
                        pbar.update(1)
        task_dirs = [d for d, ok in zip(task_dirs, mask) if ok]

        print(f"Filtering to tasks with pass@16 > 0, postfilter: {len(task_dirs)}")
        time.sleep(5)

    if cfg.use_parquet:
        from datasets import load_dataset

        dataset = load_dataset(
            "parquet", data_files=os.path.join(cfg.tasks_dir, "train.parquet")
        )["train"]
        task_dirs = [d["extra_info"]["task_dir"] for d in dataset]

    task_dirs = list(sorted(task_dirs))
    task_dirs = task_dirs[cfg.start_at : min(cfg.start_at + cfg.num_tasks, len(task_dirs))]

    if not task_dirs:
        print(f"No task directories found in {cfg.tasks_dir}")
        return

    def process_task_with_retry(task_dir: str, cfg: SolutionConfig):
        """Wrap per-task retry logic so it can run in parallel."""
        task_dir = Path(task_dir)

        sol_dir = task_dir / "solutions"
        existing = (
            [f for f in sol_dir.glob("*_summary.json") if f.name != "summary.json"]
            if sol_dir.exists()
            else []
        )
        if existing and not cfg.force_rerun:
            print(f"Skipping {task_dir.name} (already has {existing[0].name})")
            return task_dir, "skipped"
        if existing and cfg.force_rerun:
            print(f"Re-running {task_dir.name} (--force-rerun; overwriting {existing[0].name})")

        max_retries = 1
        result = None

        while max_retries > 0:
            result = process_task(task_dir, cfg)
            if result is None:
                print(f"Retrying task {task_dir.name}...")
                max_retries -= 1
            elif result in ("no def", "no sif", "no initial test"):
                print(f"No def, sif, or initial test for task {task_dir.name}, skipping.")
                break
            else:
                print(f"Pass@k: {result} for task {task_dir.name}")
                break

        return task_dir, result

    if cfg.workers <= 1:
        for task_dir in tqdm(task_dirs, desc="Processing Tasks"):
            process_task_with_retry(task_dir, cfg)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(process_task_with_retry, td, cfg): td for td in task_dirs}
            with tqdm(total=len(task_dirs), desc="Processing Tasks") as pbar:
                for fut in as_completed(futures):
                    try:
                        _td, _res = fut.result()
                    finally:
                        pbar.update(1)


def main() -> None:
    """Main entry point; optionally tee terminal output to a single log file."""
    cfg = parse_args()
    log_f: Optional[TextIO] = None
    if cfg.terminal_log:
        log_path = Path(cfg.terminal_log).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a", encoding="utf-8", errors="replace")
        log_f.write(f"\n{'=' * 80}\n")
        log_f.write(f"terminal log opened {datetime.now(timezone.utc).isoformat()}\n")
        log_f.write(f"cwd={os.getcwd()}\n")
        log_f.write(f"argv={' '.join(sys.argv)}\n")
        log_f.flush()
        sys.stdout = _TeeTextStream(sys.__stdout__, log_f)
        sys.stderr = _TeeTextStream(sys.__stderr__, log_f)
        print(f"📝 Also logging terminal output to: {log_path}", flush=True)

    try:
        _run_generate_solutions(cfg)
    finally:
        if log_f is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            try:
                log_f.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
