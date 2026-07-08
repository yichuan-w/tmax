#!/bin/bash
# ============================================================================
# TUA-Bench (120 tasks) on Daytona — tmax-9b (RL) vs base Qwen3.5-9B.
#
# WHY THIS IS FIDDLY (difficulties, all handled below):
#  1. TUA tasks BUILD from a Dockerfile at eval time (FROM ubuntu:24.04 + apt-get
#     install asciinema/ffmpeg/... ) -> ~15 min/task, and the build NEEDS internet.
#  2. Local podman can't build them: bpfjailer blocks container->internet egress
#     (apt-get fails: "Could not connect to fwdproxy"). => MUST use Daytona.
#  3. Daytona egress is confirmed OPEN (apt to ubuntu archives works), so the build
#     succeeds there. But the slow build can bleed into Harbor's agent-setup window
#     => use --agent-setup-timeout-multiplier 7.
#  4. Tasks default to cpus=6 / storage=30720 which EXCEEDS the Daytona tier ->
#     sandbox startup failures. FIX (one-time, already applied): cap to 4/8192/10240
#     on all task.toml (see cap_tua_resources() below). 61 files get capped.
#  5. Serving tmax-9b/base is the GDN-hybrid recipe: text-only (limit-mm 0), triton
#     GDN prefill, qwen3_xml tool parser, + copy preprocessor_config.json from base.
#
# PREREQ (one-time): resources capped. Re-run cap_tua_resources if tasks were reset.
# USAGE: bash run_tua_dual.sh    (serves + runs both models; self-heals; logs -> /dev/shm)
# ============================================================================
set -u
TUA=/home/yichuan/TUA-Bench/tasks
QL=/dev/shm/run_tua_dual.log
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
cd /home/yichuan/tmax
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include VerifierTimeoutError"
# TUA sandboxes are big (4cpu/10GB) + heavy build -> modest concurrency per model.
NCONC=8

cap_tua_resources(){  # cap cpus<=4 mem<=8192 storage<=10240 in every task.toml
  /home/yichuan/tmax/.venv/bin/python - <<'PY'
from pathlib import Path
caps={"cpus":4,"memory_mb":8192,"storage_mb":10240}; n=0
for p in sorted(Path("/home/yichuan/TUA-Bench/tasks").glob("*/task.toml")):
    L=p.read_text().splitlines(); out=[]; env=False; ch=False
    for ln in L:
        s=ln.strip()
        if s.startswith("[") and s.endswith("]"): env=(s=="[environment]")
        if env:
            for k,c in caps.items():
                if s.startswith(f"{k} ="):
                    b,v=ln.split("=",1)
                    try: cur=int(v.strip())
                    except: break
                    if cur>c: ln=f"{b}= {c}"; ch=True
                    break
        out.append(ln)
    if ch: p.write_text("\n".join(out)+"\n"); n+=1
print("capped",n,"task.toml")
PY
}

pyget(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['stats']['n_completed_trials'])" 2>/dev/null; }
pytot(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['n_total_trials'])" 2>/dev/null; }

heal(){  # keep both GDN serves up (self-heal on crash)
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8016/v1/models)" != 200 ] && ! pgrep -f 'serve_tmax9b_tua.sh' >/dev/null && { echo "$(date '+%T') !! tmax serve down" >>"$QL"; setsid bash /home/yichuan/tmax/serve_tmax9b_tua.sh >/dev/null 2>&1 </dev/null & }
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8017/v1/models)" != 200 ] && ! pgrep -f 'serve_base9b_tua.sh' >/dev/null && { echo "$(date '+%T') !! base serve down" >>"$QL"; setsid bash /home/yichuan/tmax/serve_base9b_tua.sh >/dev/null 2>&1 </dev/null & }
  pgrep -f tua_daemon.py >/dev/null || setsid /home/yichuan/tmax/.venv/bin/python tua_daemon.py >/dev/null 2>&1 </dev/null &
}
runeval(){ # port model job
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run -p "$TUA" --env daytona --yes \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent $NCONC -k 1 --agent-setup-timeout-multiplier 7 $RETRY \
    --job-name $3 > /dev/shm/eval_$3.log 2>&1 </dev/null & }

echo "$(date '+%F %T') TUA dual run start" > "$QL"
cap_tua_resources >> "$QL" 2>&1
# start serves ONLY if down (heal is idempotent — avoids double-starting on busy GPUs)
heal
for i in $(seq 1 30); do { [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8016/v1/models)" = 200 ] && [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8017/v1/models)" = 200 ]; } && break; sleep 15; done
echo "$(date '+%T') both serves up; launching TUA on both" >> "$QL"

rm -rf jobs/tua-tmax jobs/tua-base 2>/dev/null
runeval 8016 tmax-9b        tua-tmax
runeval 8017 qwen35-9b-base tua-base

# monitor + self-heal until both done (or near-done + stalled)
last=-1; stall=0
for i in $(seq 1 20000); do
  heal
  a=$(pyget tua-tmax); b=$(pyget tua-base); ta=$(pytot tua-tmax); tb=$(pytot tua-base)
  a=${a:-0}; b=${b:-0}; ta=${ta:-120}; tb=${tb:-120}; sum=$((a+b))
  if [ "$sum" = "$last" ]; then stall=$((stall+1)); else stall=0; last=$sum; fi
  echo "$(date '+%T') tua-tmax=$a/$ta tua-base=$b/$tb stall=$stall" >> "$QL"
  { [ "$a" -ge "$ta" ] && [ "$b" -ge "$tb" ] && [ "$ta" -gt 0 ]; } && { echo "$(date '+%T') === TUA DUAL DONE ===" >> "$QL"; break; }
  { [ "$stall" -ge 15 ] && [ "$a" -ge $((ta-5)) ] && [ "$b" -ge $((tb-5)) ] && [ "$ta" -gt 100 ]; } && { echo "$(date '+%T') === TUA near-done+stalled DONE ===" >> "$QL"; break; }
  sleep 60
done
