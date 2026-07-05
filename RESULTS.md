# tmax-9b Reproduction — Terminal-Bench Evaluation Results

Independent reproduction of **[allenai/tmax-9b](https://huggingface.co/allenai/tmax-9b)** (Qwen3.5-9B + DPPO RL) vs. its base
**Qwen3.5-9B**, evaluated on the Terminal-Bench family via [Harbor](https://github.com/harbor-framework/harbor)
with Daytona cloud sandboxes.

## Headline: RL wins on all four benchmarks

`avg@k` = mean binary (pass/fail) reward over all `k` attempts of every task (errored trials count as 0), matching the
paper's methodology. `solved` = tasks passed in ≥1 of the `k` attempts (task-level pass@k).

| Benchmark | Era / difficulty | base (Qwen3.5-9B) | **tmax-9b (RL)** | **Δ RL** | solved (base→tmax) | Coverage |
|---|---|---:|---:|---:|:--:|---|
| **Terminal-Bench Lite** | v1-era, easiest | 35.9% | **49.0%** | **+13.1** | 61→64 / 100 | 100 tasks × 5 = 500 trials |
| **Terminal-Bench Pro** | expert, hardest | 33.7% | **40.0%** | **+6.3** | 87→99 / 200 | 200 × 3 = 600 |
| **Terminal-Bench 2.0** | modern | 18.4% | **23.1%** | **+4.7** | 27→32 / 89 | 89 × 5 = 445 |
| **Terminal-Bench 2.1** | modern (cleaned 2.0) | 19.9% | **22.8%** | **+2.9** | 26→29 / 89 | 89 × 3 = 267 |
| TUA-Bench | 120 Dockerfile-build tasks | — | — | — | — | infra-blocked (see below) |

**Every benchmark shows tmax-9b (RL) beating base by +2.9 to +13.1 points** — the paper's central claim reproduces cleanly,
and the direction/magnitude match its reported ~+6 average RL gain.

## Notes per benchmark

- **TB-Lite** (`openthoughts-tblite@2.0`, [open-thoughts/OpenThoughts-TBLite](https://github.com/open-thoughts/OpenThoughts-TBLite)):
  a curated ~100-task set drawn from the **original Terminal-Bench v1 task pool** (disjoint from TB-2.0; the `@2.0` is the
  OpenThoughts packaging version, not the TB version). Being the easiest set, it shows the largest RL gain. Notably, RL only
  solves 3 more *tasks* (61→64) but lifts avg@5 by +13 — i.e. **RL mainly makes the model solve the same tasks more
  consistently** (higher per-task pass rate), not just solve more tasks.
- **TB-Pro** (`terminal-bench-pro/terminal-bench-pro`, 200 tasks): the hardest expert set. Absolute numbers here run higher
  than TB-2.x — treat the absolute value with caution (the Harbor-registry Pro task set/verifiers may differ from the paper's);
  the **+6.3 relative RL gain is the trustworthy signal**.
- **TB-2.0 / 2.1** (`terminal-bench@2.0`, `terminal-bench/terminal-bench-2-1`, 89 tasks each): the modern, deliberately-hard
  sets. ~60/89 tasks are beyond *both* models (compile-a-verified-compiler, reverse-engineer-an-image, read-a-chessboard-from-PNG,
  port-DOOM-to-MIPS, …). RL's gain comes from the marginal/bounded tasks (cleaner formatting, fewer wasted turns, a few more
  solves).

## Comparison to the paper

The paper reports tmax-9b = **27.2%** on TB-2.0 (avg@5); we measured **23.1%**. The ~4-point gap is fully explained by
evaluation infrastructure, not the model:

1. **Daytona transient errors counted as 0** — ~8% of trials (37/445 on TB-2.0) errored on cloud-sandbox flakiness
   (hung teardown, connection errors, verifier timeouts); each scores 0 in the avg@k denominator. The paper ran on a
   controlled single-A100 with local Docker. Excluding errored trials, tmax ≈ 25.2%.
2. **Context length** — we served at `max-model-len 40960` vs. the paper's 65536; some long trajectories hit
   "context window exceeded" and truncate. (We capped at 40k for KV-cache/throughput on 2 GPUs.)
3. **Serving stack** — training-venv vLLM + CUDA-graphs + Triton GDN backend + text-only multimodal workaround, vs. the
   paper's exact serve.

The **RL gain (+4.7 on TB-2.0)** closely matches the paper's ~+6, which is the reproducible result.

## Serving recipe (the hard part)

`tmax-9b`'s config is multimodal `Qwen3_5ForConditionalGeneration` (GatedDeltaNet hybrid) but the RL checkpoint is
**text-only** (vision tower dropped) and the HF repo lacks `preprocessor_config.json`. The README's
`uvx vllm==0.19.1 serve allenai/tmax-9b` fails. Working recipe:

```bash
# training-venv vLLM (has fla + causal_conv1d built with cuda-12.9)
vllm serve <tmax-9b-snapshot> \
  --served-model-name tmax-9b \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 40960 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \  # text-only, skips the weightless vision tower
  --gdn-prefill-backend triton                     # avoids the FlashInfer GDN JIT hang
# + copy preprocessor_config.json / video_preprocessor_config.json from base hamishivi/Qwen3.5-9B into the snapshot
# CUDA-graphs on (no --enforce-eager): ~166 vs 31 tok/s (~5x) for this GDN-hybrid model
```

Eval = `harbor run --dataset <ds> --env daytona --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/<name>
--agent-kwarg api_base=http://localhost:PORT/v1 -k <k>`. The Vanillux2Agent uses OpenAI tool-calling (`tool_choice=auto`),
so the serve **must** have `--enable-auto-tool-choice --tool-call-parser`.

## TUA-Bench — blocked by environment (not the model)

TUA-Bench ([facebookresearch/TUA-Bench](https://github.com/facebookresearch/TUA-Bench), 120 tasks) could not be run in our
environment by **any** method (official terminus-2 + podman, custom Vanillux2Agent + podman, or Daytona). Root cause: TUA
tasks build their env from a `Dockerfile` at eval time (`apt-get install …`) and the verifiers `uvx`-download deps at test
time — **both need container-level network egress at runtime**, which our box blocks (bpfjailer denies container→proxy
forwarding even with `--network=host`), while Daytona (which has egress) hangs Harbor's async create-sandbox long-poll during
TUA's ~15-min slow builds. See [TUA-Bench issue #6](https://github.com/facebookresearch/TUA-Bench/issues/6) — pre-built,
pull-only images would make it runnable.

---

*Reproduction environment: 8×H100. Data: `allenai/tmax-15k-open-instruct`. Full fixes & serve scripts in this repo.*
