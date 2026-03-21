"""Generate lightweight per-task Apptainer defs on top of pre-built domain base images."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
import re
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from rl_data import chat_completion_batch, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Base image registry: domain -> .sif path
# ---------------------------------------------------------------------------

CONTAINERS_DIR = Path(__file__).resolve().parent.parent / "containers"

BASE_IMAGES: dict[str, Path] = {
    "security":              CONTAINERS_DIR / "base_security.sif",
    "software_engineering":  CONTAINERS_DIR / "base_software_engineering.sif",
    "file_operations":       CONTAINERS_DIR / "base_file_operations.sif",
    "data_querying":         CONTAINERS_DIR / "base_data_querying.sif",
    "data_science":          CONTAINERS_DIR / "base_data_science.sif",
    "debugging":             CONTAINERS_DIR / "base_debugging.sif",
    "scientific_computing":  CONTAINERS_DIR / "base_scientific_computing.sif",
    "data_processing":       CONTAINERS_DIR / "base_data_processing.sif",
    "system_administration": CONTAINERS_DIR / "base_system_administration.sif",
}

DEFAULT_BASE = CONTAINERS_DIR / "base_software_engineering.sif"


def _resolve_base(domain: str) -> Path:
    base = BASE_IMAGES.get(domain, DEFAULT_BASE)
    if not base.exists():
        base = DEFAULT_BASE
    return base


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_MSG = """\
You are an expert in Apptainer/Singularity container setup.

You will be given a task description, ground truth, and initial-state tests.
Your job is to write an Apptainer .def file that sets up the initial state
of the container so that an agent can be tested on the task.

IMPORTANT RULES:
- Always start the def file with exactly:
  Bootstrap: docker
  From: ubuntu:22.04
- In the %post section:
  1. Start with: export DEBIAN_FRONTEND=noninteractive
  2. Run: apt-get update && apt-get install -y python3 python3-pip
  3. Run: pip3 install pytest
  4. Install ONLY the additional system or Python packages the task needs.
     Keep package installs minimal. Prefer pip over apt when possible.
  5. Create files, directories, and data needed for the task.
  6. Create the user: useradd -m -s /bin/bash user || true
  7. End with: chmod -R 777 /home/user
- Do NOT include %test sections.
- Do NOT create output files that the agent should produce.
- The home path is /home/user.
- Do NOT override HOME in %environment.
- Do NOT use Apptainer build variables (no {{ }}).
- Do NOT use exotic package names. If you need awk, install gawk.
  The command 'tr' is part of coreutils, not a separate package."""

BASE_USER_TEMPLATE = """\
Write an Apptainer .def file for this task.

The task domain is: {domain}

Task description given to the agent:
{task_description}

Ground truth (for setting up initial state):
{truth}

Tests that will verify the initial container state:
{test_py}

Previous failures (may be empty):
{failures}

Respond with ONLY the Apptainer .def file. It must start with:
Bootstrap: docker
From: ubuntu:22.04

Keep the %post section focused: install only what's needed, create the
required files/directories/data, and ensure /home/user is writable."""


def build_and_test(def_template: str, test_py: str) -> tuple[bool, str]:
    """Build an Apptainer image from a def and run initial-state tests."""
    import os
    tmp_base = os.environ.get("APPTAINER_TMPDIR", None)
    with tempfile.TemporaryDirectory(dir=tmp_base) as td:
        td_path = Path(td)

        def_path = td_path / "container.def"
        def_path.write_text(def_template)

        test_file = td_path / "test_initial_state.py"
        test_file.write_text(test_py)

        sif_path = td_path / "img.sif"
        build_proc = subprocess.run(
            ["apptainer", "build", str(sif_path), str(def_path)],
            capture_output=True, text=True, timeout=300,
        )
        if build_proc.returncode:
            err_snippet = (build_proc.stderr or build_proc.stdout or "")[-500:]
            print(f"Apptainer build failed (rc={build_proc.returncode}): {err_snippet}")
            return False, f"Apptainer build failed: {err_snippet}"

        proc = subprocess.run(
            [
                "apptainer", "exec",
                "--fakeroot", "--userns", "--writable-tmpfs", "--cleanenv",
                str(sif_path),
                "pytest", "-q", str(test_file.name),
            ],
            cwd=td,
            capture_output=True, text=True,
        )

        if sif_path.exists():
            sif_path.unlink()
        shutil.rmtree(td_path, ignore_errors=True)

        return proc.returncode == 0, proc.stdout + proc.stderr


def parse_def_template(def_template: str) -> str:
    """Extract and clean a .def file from LLM output."""
    cleaned = def_template.replace("\r\n", "\n").strip()

    fence_re = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\n(?P<code>[\s\S]*?)```", re.MULTILINE)
    match = fence_re.search(cleaned)
    if match:
        cleaned = match.group("code").strip()

    cleaned = textwrap.dedent(cleaned).strip()
    return cleaned


def iterate_def_template_batch(
    items: List[Tuple[str, str, str]],
    *,
    domains: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    max_concurrency: int = 64,
) -> List[Optional[str]]:
    """Batched def generation with pre-built base images.

    Parameters
    ----------
    items : list of (task_description, truth, test_py)
    domains : list of domain strings aligned with items (selects the base image)
    """
    if domains is None:
        domains = ["software_engineering"] * len(items)

    messages: list[list[dict[str, str]]] = []
    for (task_description, truth, test_py), domain in zip(items, domains):
        prompt = BASE_USER_TEMPLATE.format(
            domain=domain.replace("_", " "),
            task_description=task_description,
            truth=truth,
            test_py=test_py,
            failures="None yet",
        )
        messages.append([
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ])

    responses = chat_completion_batch(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        num_completions=1,
        max_concurrency=max_concurrency,
    )

    results: List[Optional[str]] = [None] * len(items)

    def worker(index: int, item: Tuple[str, str, str], resp_obj) -> Tuple[int, Optional[str]]:
        try:
            if resp_obj is None:
                return index, None
            content = resp_obj.choices[0].message.content
            def_text = parse_def_template(content)
            _task_description, _truth, test_py = item
            ok, _ = build_and_test(def_text, test_py)
            return index, (def_text if ok else None)
        except Exception:
            return index, None

    build_workers = min(4, len(items))
    futures = []
    with ThreadPoolExecutor(max_workers=build_workers) as executor:
        for idx, (item, resp) in enumerate(zip(items, responses)):
            futures.append(executor.submit(worker, idx, item, resp))

        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx, value = fut.result()
            results[idx] = value

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-path", type=str, default="tasks/sample_task")
    args = ap.parse_args()
    task_path = Path(args.task_path)
    def_path = task_path / "container.def"
    initial_test_path = task_path / "test_initial_state.py"

    test_py = initial_test_path.read_text()
    def_text = def_path.read_text()

    success, output = build_and_test(def_text, test_py)
    print(success)
    print(output)
