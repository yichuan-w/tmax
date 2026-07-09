# Running TUA-Bench on Daytona (Harbor) — the fix

If you try to evaluate **TUA-Bench** (120 tasks, aka `arxiv_v1`) through Harbor with
`--env daytona` and it seems to **hang for ~15 min per task with no progress**, this is
almost certainly **not a slow Docker build** — it's a **sandbox setup hang**. Below is the
root cause and the exact fixes.

## TL;DR

1. TUA task Dockerfiles end with `USER agent` (non-root). The Daytona sandbox therefore runs
   as `agent`, which has **no sudo** and **cannot create/write `/tests`, `/solution`, `/logs`**.
2. Harbor's post-build setup writes to those dirs and runs some commands as root via
   **`su root`**, which prompts for a password and **hangs forever**.
3. Fix = (a) cap task resources to the Daytona tier, and (b) patch Harbor's Daytona backend to
   install sudo + pre-create/chown those dirs at build time, and to elevate via `sudo -n`
   instead of `su root`.

With both fixes, a task **builds + sets up + runs the agent in ~15–20 s** (Daytona caches the
image layers). TUA is fast, not slow.

## Symptom / how to confirm the root cause

The Harbor log stops right after `Building environment from .../Dockerfile` and goes silent,
while `jobs/<name>/result.json` stays at `0/N`. But the Daytona sandbox is actually **STARTED**
(built fine). Exec into it:

```python
from daytona import Daytona, DaytonaConfig
# ... build client (proxy-patch Configuration if behind a corp proxy) ...
sb = [s for s in d.list() if (s.labels or {}).get("tmaxeval")=="1" and "STARTED" in str(s.state)][0]
print(sb.process.exec("whoami; id; ls -ld /tests /solution /logs; sudo -n true || echo NO_SUDO").result)
```

You'll see `USER=agent`, `sudo: command not found`, and `/tests /solution /logs: No such file`.
That's the hang: Harbor can't set up the sandbox as `agent`.

> Note: TB (Terminal-Bench 2.0/Lite/Pro) does **not** hit this — its task Dockerfiles don't set
> a non-root final `USER`, so Harbor's setup runs as root fine. Both TB and TUA *build* from a
> Dockerfile (neither uses a prebuilt `docker_image` pull); TUA is only "slow" because of this hang.

## Fix 1 — cap task resources to the Daytona tier

TUA task.toml default to `cpus=6` / `storage_mb=30720`, which exceeds common Daytona tiers and
causes sandbox-startup failures. Cap them (idempotent; ~61 files change):

```python
from pathlib import Path
caps = {"cpus": 4, "memory_mb": 8192, "storage_mb": 10240}
for p in sorted(Path("TUA-Bench/tasks").glob("*/task.toml")):
    L = p.read_text().splitlines(); out=[]; env=False; ch=False
    for ln in L:
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"): env = (s == "[environment]")
        if env:
            for k, c in caps.items():
                if s.startswith(f"{k} ="):
                    b, v = ln.split("=", 1)
                    try: cur = int(v.strip())
                    except: break
                    if cur > c: ln = f"{b}= {c}"; ch = True
                    break
        out.append(ln)
    if ch: p.write_text("\n".join(out) + "\n")
```

## Fix 2 — patch Harbor's Daytona backend

These live in the installed package and are **not tracked by git** — reapply after any
`uv sync` / venv rebuild. File:
`.venv/lib/python3.12/site-packages/harbor/environments/daytona.py`

**(a) Add two helpers** (module level, after the imports):

```python
def _final_dockerfile_user(dockerfile_path):
    user = None
    for line in Path(dockerfile_path).read_text().splitlines():
        s = line.strip()
        if s.upper().startswith("USER "):
            user = s.split(maxsplit=1)[1]
    return user

def _add_daytona_sudo_support(image, dockerfile_path):
    final_user = _final_dockerfile_user(dockerfile_path)
    commands = [
        "USER root",
        "RUN set -eux; "
        "if command -v apt-get >/dev/null 2>&1; then apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y sudo && rm -rf /var/lib/apt/lists/*; "
        "elif command -v apk >/dev/null 2>&1; then apk add --no-cache sudo; "
        "elif command -v dnf >/dev/null 2>&1; then dnf install -y sudo; "
        "elif command -v yum >/dev/null 2>&1; then yum install -y sudo; fi; "
        "mkdir -p /logs/agent /logs/verifier /logs/artifacts /tests /solution; "
        "if id agent >/dev/null 2>&1; then "
        "  if command -v usermod >/dev/null 2>&1; then (usermod -aG sudo agent 2>/dev/null || usermod -aG wheel agent 2>/dev/null || true); fi; "
        "  if command -v sudo >/dev/null 2>&1; then mkdir -p /etc/sudoers.d; echo 'agent ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/agent; chmod 0440 /etc/sudoers.d/agent; fi; "
        "  chown -R agent:agent /logs /tests /solution; "
        "else chmod -R 777 /logs /tests /solution; fi",
    ]
    if final_user:
        commands.append(f"USER {final_user}")
    return image.dockerfile_commands(commands)
```

**(b) Wrap the Dockerfile build** so the sandbox gets sudo + the dirs:

```python
# was: image = Image.from_dockerfile(env._dockerfile_path)
image = _add_daytona_sudo_support(
    Image.from_dockerfile(env._dockerfile_path), env._dockerfile_path,
)
```

**(c) Fix `_sandbox_exec` to elevate via `sudo -n`, never the hanging `su root`:**

```python
if user is not None:
    user_arg = (f"$(getent passwd {user} | cut -d: -f1)" if isinstance(user, int)
                else shlex.quote(user))
    quoted = shlex.quote(command)
    if user == "root" or user == 0:
        command = ('if [ "$(id -u)" -eq 0 ]; then ' + command +
                   "; else sudo -n bash -c " + quoted + "; fi")
    else:
        command = ("target_user=" + user_arg + '; target_uid="$(id -u "$target_user")"; '
                   'if [ "$(id -u)" = "$target_uid" ]; then ' + command +
                   '; elif [ "$(id -u)" -eq 0 ]; then su "$target_user" -s /bin/bash -c ' + quoted +
                   "; else sudo -n -u \"$target_user\" bash -c " + quoted + "; fi")
```

Verify: `python -m py_compile .venv/.../harbor/environments/daytona.py`.

## How to run

Serve the model(s) with the GDN recipe (text-only, triton GDN, `qwen3_xml`), then run Harbor
with `--force-build` (so the sudo layer is baked in) and a bumped agent-setup timeout (heavy
builds bleed into the setup window):

```bash
uv run python eval_harbor.py run -p TUA-Bench/tasks --env daytona --yes --force-build \
  --agent-import-path Vanillux2Agent:Vanillux2Agent \
  --model openai/tmax-9b --agent-kwarg api_base=http://localhost:8016/v1 \
  --agent-kwarg max_format_errors=64 --n-concurrent 20 -k 5 \
  --agent-setup-timeout-multiplier 7 \
  --max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaError \
  --retry-include DaytonaNotFoundError --retry-include VerifierTimeoutError \
  --job-name tua-tmax
```

`-k 5` runs each of the 120 tasks 5× (= 600 trials) for **avg@5**. Two model serves (tmax-9b on
`:8016`, base on `:8017`) run concurrently; `run_tua_dual.sh` / `_tua_tmax.sh` / `_tua_base.sh` in
this repo wrap all of the above (resource caps + self-healing GDN serves + both models via Daytona).

> **Tail rate-limit:** running both evals at once (2 × `--n-concurrent 20`) can throttle the Daytona
> control plane near the end (`DaytonaAuthorizationError` / `ThrottlerException` / `DaytonaRateLimitError`),
> spiking the errored-trial count. Those are infra, not model failures — report avg@5 **excluding
> errored trials** for a fair model comparison, or lower `--n-concurrent` / stagger the two runs.

## Result (this repo's run — tmax-9b RL vs base Qwen3.5-9B, k=5)

| Metric | base (Qwen3.5-9B) | **tmax-9b (RL)** | Δ RL |
|---|---:|---:|---:|
| completed | 597 / 600 | **600 / 600** | |
| **avg@5 (error-excluded)** | **16.4%** | **24.9%** | **+8.5** |
| avg@5 (raw, errored = 0) | 10.2% | 19.1% | +8.9 |
| perfect solves (reward = 1.0) | 56 | **91** | +35 |
| tasks solved (≥1 positive attempt) | 40 / 120 | **50 / 120** | +10 |

TUA rewards are continuous (0–1 partial credit). RL wins by **+8.5 avg@5** — the recipe's
second-largest 9B gain, as expected for its own training domain. See `RESULTS.md` for the full
Terminal-Bench-family comparison.

## Gotchas

- **`--force-build`** is required so the sudo/dirs layer is rebuilt into the image.
- **Don't `pkill -f eval_harbor.py`** in monitoring scripts — it kills your own run. Launch each
  eval detached (`setsid bash <script> </dev/null &`) and check without pkilling.
- **Daytona egress must be on** (build does `apt-get`). If your account is restricted, redeem the
  `HARBOR_NETWORK` coupon in the Daytona dashboard. (Local podman can't build these — bpfjailer
  blocks container→internet — so Daytona is required.)
- These `.venv` patches are not in git; reapply after any dependency reinstall.
