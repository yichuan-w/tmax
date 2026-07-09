# TMAX Evaluation Campaign — Process & Conclusions

Full log of an evaluation session on an 8×H100 box: reproducing the TMAX paper's RL gains
(9B + 27B) and evaluating the user's **own** locally-trained Qwen3.5-9B RL checkpoints
(step-40, step-100). Eval stack = [Harbor](https://github.com/harbor-framework/harbor) +
Daytona cloud sandboxes, `Vanillux2Agent`, `avg@k` = mean pass/fail reward over all `k`
attempts (errored trials = 0), matching the paper's methodology. See `RESULTS.md` for the
9B/27B reproduction write-up; this doc adds the user's checkpoint evals + infra findings.

---

## 1. Headline conclusions

1. **RL gain is a small-model phenomenon.** At **9B**, DPPO on TMAX-15K wins by **+2.9…+13.1**
   across the Terminal-Bench family, **plus +8.5 on TUA-Bench** (the recipe's own terminal-agent
   training domain, k=5 avg@5). At **27B** it collapses to **≤+0.9 on every benchmark** — the
   strong Qwen3.6-27B base is already near the recipe's ceiling. This reproduces the paper's own
   admission ("the gap is biggest for TMAX-9B … [27B] much harder to improve").
2. **The user's own RL run at step-100 shows no gain yet** — it tracks the **base** model on all
   four benchmarks (≈ within noise), well below the official `tmax-9b`. step-40 had actually
   dipped slightly *below* base; step-100 recovered *to* base. Interpretation: either too early
   (official tmax-9b trained hundreds of steps) or the reduced single-node config under-delivers.
   Next diagnostic: the training reward curve, or eval a later checkpoint (step-160/240).
3. **Daytona sandbox egress is real and TB scores are valid** w.r.t. networking (verified live).

---

## 2. 27B reproduction — TMAX-27B vs Qwen3.6-27B  (avg@k, errors=0)

| Benchmark | base (Qwen3.6-27B) | tmax-27b (RL) | **Δ ours** | Δ paper (Table 3) |
|---|---:|---:|---:|---:|
| TB-Lite | 63.0% | 63.7% | **+0.6** | −2.2 |
| TB-2.0  | 33.5% | 34.4% | **+0.9** | (not reported) |
| TB-2.1  | 34.8% | 35.2% | **+0.4** | +4.4 |
| TB-Pro  | 47.7% | 48.0% | **+0.3** | (not reported) |

RL gain ≤1 pt everywhere. TB-2.1's *direction* matches the paper's +4.4 claim but not the
*magnitude* (ours +0.4); the paper's own gap is ~1.5σ. Absolute levels run below the paper's
(TB-Lite base 63.0 vs 70.8) due to eval infra (Daytona errors=0, max-model-len 40960 vs 65536,
serving stack) — so the **RL delta** is the trustworthy signal. Full detail in `RESULTS.md`.

## 3. 9B reference — official tmax-9b vs base  (from `RESULTS.md`)

| Benchmark | base (Qwen3.5-9B) | tmax-9b (RL) | **Δ** |
|---|---:|---:|---:|
| TB-Lite | 35.9% | 49.0% | **+13.1** |
| TUA-Bench (k5) † | 16.4% | 24.9% | **+8.5** |
| TB-Pro  | 33.7% | 40.0% | **+6.3** |
| TB-2.0  | 18.4% | 23.1% | **+4.7** |
| TB-2.1  | 19.9% | 22.8% | **+2.9** |

† TUA-Bench avg@5 shown **error-excluded** (raw errored-as-0 = 19.1 vs 10.2, +8.9). Its error rate
ran high (tmax 25% / base 43%) from a shared-Daytona rate-limit throttle at the tail, not model
failures — the RL delta is robust either way. TUA rewards are continuous (partial credit); tmax
also solved more tasks (50 vs 40 / 120) and hit 91 vs 56 perfect-solve trials. See `RESULTS.md`
and `TUA_BENCH_DAYTONA.md`.

## 4. User's own checkpoints — step-40 & step-100  (torchtitan Qwen3.5-9B + RL)

avg@k shown as **raw (errors=0) / error-excluded**. base-9b & tmax-9b columns are raw (from §3).

| Benchmark | base-9b | **step-40** | **step-100** | tmax-9b (official) |
|---|---:|---:|---:|---:|
| TB-Lite (k5) | 35.9 | — | **35.4 / 39.2** | 49.0 |
| TB-2.0 (k5)  | 18.4 | **17.3 / 18.6** | **18.0 / 20.5** | 23.1 |
| TB-2.1 (k3)  | 19.9 | — | **18.7 / 20.2** | 22.8 |
| TB-Pro (k3)  | 33.7 | — | **32.8 / 34.4** | 40.0 |

**Verdict: step-100 ≈ base on all four benchmarks (within ~1 pt), no measurable RL gain.**
Trajectory on TB-2.0: base 18.4 → step-40 17.3 (dip) → step-100 18.0 (back to base). The run
is nowhere near the official tmax-9b (+5–14 pts). Not a serving bug — the model serves and
generates coherently; it simply hasn't learned the recipe's gains by step-100.

---

## 5. Infrastructure findings & fixes (the hard part)

### 5.1 Daytona egress verified
TB-2.0/2.1 verifiers need public internet (50+ tasks' `test.sh` run `apt-get` /
`curl -LsSf https://astral.sh/uv/install.sh | sh` / `uvx`). Confirmed: harbor sets
`network_block_all = not task_env_config.allow_internet`; `allow_internet` defaults **True** and
no TB task overrides it. Live test in an ephemeral sandbox ran the exact uv installer →
"everything's installed!" (uv 0.11.26), pypi HTTP 200, github 301, apt OK. Non-zero eval scores
independently confirm egress worked. (Caveat: local *podman training* sandboxes block egress via
bpfjailer, but training tasks need 0 network — unrelated.)

### 5.2 torchtitan DCP → HF conversion
The user's checkpoints are **torchtitan** DCP (`__N_0.distcp` shards + `.metadata`), keys in
torchtitan style (`tok_embeddings`, `layers.N.attn.{wq…,in_proj_*,conv_*,A_log,dt_bias}`,
`feed_forward.w1/w2/w3`), **including a vision encoder** (full multimodal Qwen3.5-9B GDN hybrid).
Converted via `manual_convert_step100.py` (torchtitan venv): build `model_registry("9B")`, load
DCP, `state_dict_adapter.to_hf()`, save bf16 safetensors directly. Weights scanned clean
(0 NaN/Inf).

### 5.3 GDN serve crash → `--cudagraph-mode FULL_DECODE_ONLY` (fast **and** stable)
The vLLM serve worker died silently under concurrent load (GPU-side illegal access in the FLA/
triton GatedDeltaNet **prefill** kernel — not OOM, not bad weights; crashed at KV 2.4%). step-40
crashed ~1× over its run; step-100 crashed every ~2–3 min even at concurrency 6.
- `FULL_AND_PIECEWISE` (default): crashes constantly.
- `--enforce-eager`: stable, but ~2× slower.
- **`--cudagraph-mode FULL_DECODE_ONLY`**: graphs only pure-decode (stable shape) + runs prefill
  eager (the part that crashed) → **stable AND ~2× faster than enforce-eager**. This is the fix.

### 5.4 Throughput: DP=2 × TP=4 behind a round-robin proxy
GPU was compute-bound (100% util) at concurrency 12 on a single TP=4 serve — raising concurrency
alone did nothing. Solution: two TP=4 serves (GPU 2,3,4,5 and 0,1,6,7) behind a tiny round-robin
proxy (`lb_proxy.py`, :8020), harbor at `--n-concurrent 48`. Result: **~13 trials/min, ~6× the
original** single-serve/low-concurrency rate. (9B is small → DP scales better than TP=8.)

### 5.5 Daytona billing backstop
Shared Daytona key with a co-tenant (`titan_swe_r2e`) — never touch theirs. Harbor creates
sandboxes with native idle-auto-stop **disabled** (`auto_stop_interval=0`), so cleanup relied
only on ephemeral-delete + a local orphan daemon (both die if the box dies). Fixed by having
`tua_daemon.py` set Daytona-native `auto_stop_interval=30` on every `tmaxeval` sandbox each cycle
→ Daytona itself stops+deletes idle orphans (with `auto_delete_interval=0` = delete-on-stop),
bounding billing even if all local scripts die.

### 5.7 TUA-Bench "slow build" was a **setup hang** → Daytona backend patch
TUA looked un-runnable (each task seemed to "hang ~15 min", `result.json` stuck at 0/N). It is
**not** a slow build: the sandbox builds + STARTS in seconds. Root cause — TUA task Dockerfiles end
with `USER agent` (non-root), so the Daytona sandbox runs as `agent`, which has **no sudo** and
cannot create/write `/tests` `/solution` `/logs`. Harbor's post-build setup writes those and runs
some commands as root via **`su root`** → password prompt → hangs forever. (TB tasks don't set a
non-root final `USER`, so they run setup as root and never hit this.) Fix (full recipe in
**`TUA_BENCH_DAYTONA.md`**): (a) cap task resources to the Daytona tier (cpus≤4 / mem≤8192 /
storage≤10240 — defaults of 6/30720 exceed it and fail sandbox startup); (b) patch
`.venv/…/harbor/environments/daytona.py` — at build time install sudo, pre-create + chown
`/tests //solution //logs`, grant `agent` NOPASSWD sudo, restore the final `USER`; and elevate in
`_sandbox_exec` via **`sudo -n bash -c`** (never the hanging `su root`). Serve with `--force-build`
so the sudo layer bakes in. Result: a task builds + sets up + runs in **~15–20 s** — TUA is fast,
not slow. (These `.venv` patches aren't git-tracked; backup at `/dev/shm/daytona.py.bak`, reapply
after any `uv sync`.) The tail rate-limit error spike (25%/43%) is a separate, transient effect of
two evals hammering the Daytona control plane at once, not the setup hang.

### 5.6 Disk / quota workarounds
`/home` sits at its per-user disk quota; eval log/job churn repeatedly tripped ENOSPC (which
freezes command-output capture). Mitigations: serve logs → `/dev/shm`, HF weights served from
`/dev/shm` (frees /home + faster load), and periodic cleanup of completed job dirs + old serve
logs. When output capture froze: minimal/suppressed commands still execute; kill GPU procs by
explicit PID via `nvidia-smi --query-compute-apps=pid`.

---

## 6. Reusable artifacts in this repo

| File | Purpose |
|---|---|
| `manual_convert_step100.py` | torchtitan DCP → HF safetensors converter (step-N) |
| `serve_step100_cg.sh` / `serve_step100_cg2.sh` | TP=4 FULL_DECODE_ONLY serve (GPU 2-5 / 0-7-etc) |
| `lb_proxy.py` | round-robin proxy → DP over two serves |
| `run_step100_bench2.sh` | dual-serve self-healing bench suite (TB-Lite→2.1→Pro) |
| `run_27b_otherbench.sh` | 27B self-healing bench queue |
| `tua_daemon.py` | Daytona orphan cleaner + native auto-stop arming |
| `run_tua_dual.sh` / `_tua_tmax.sh` / `_tua_base.sh` | TUA-Bench dual run (tmax-9b vs base), self-healing serves |
| `serve_tmax9b_tua.sh` / `serve_base9b_tua.sh` | GDN serve recipes for TUA (tmax :8016 / base :8017) |
| `TUA_BENCH_DAYTONA.md` | how to run TUA-Bench on Daytona — the setup-hang fix |
| `eval_harbor.py` | Harbor eval wrapper |

*Environment: 8×H100. Data: `allenai/tmax-15k-open-instruct`. Models: `hamishivi/Qwen3.5-9B`,
`Qwen/Qwen3.6-27B`, `allenai/tmax-9b`, `allenai/tmax-27b`, + user's torchtitan step-N checkpoints.*
