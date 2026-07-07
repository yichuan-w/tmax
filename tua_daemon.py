#!/usr/bin/env python
"""Orphan-cleanup daemon for the low-concurrency TUA run. Deletes ONLY sandboxes we
created (label tmaxeval=1) that are orphaned — NEVER touches the co-tenant torchtitan
`owner=titan_swe_r2e` RL run sharing this Daytona key. Orphan = tmaxeval=1 AND
(BUILD_FAILED/ERROR any age  OR  STARTED/CREATING older than 80 min = past any legit
trial lifetime of build~15m + agent<=40m + verify~5m)."""
import os, time, datetime as dt, concurrent.futures as cf
from collections import Counter
from daytona import Daytona, DaytonaConfig
import daytona_api_client.configuration as _c
_orig = _c.Configuration.__init__
def _p(self, *a, **k):
    _orig(self, *a, **k); self.proxy = os.environ.get("https_proxy")
_c.Configuration.__init__ = _p
d = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))

def _age_min(s):
    try:
        ca = dt.datetime.fromisoformat(str(s.created_at).replace("Z", "+00:00"))
        return (dt.datetime.now(dt.timezone.utc) - ca).total_seconds() / 60.0
    except Exception:
        return 0.0

def _is_orphan(s):
    if (s.labels or {}).get("tmaxeval") != "1":
        return False  # not ours -> NEVER delete
    st = str(getattr(s, "state", ""))
    if "BUILD_FAILED" in st or "ERROR" in st:
        return True
    if ("STARTED" in st or "CREATING" in st) and _age_min(s) > 80:
        return True
    return False

# Daytona-NATIVE billing backstop: enable idle auto-stop on our sandboxes so Daytona itself
# stops (and, with auto_delete_interval=0, immediately deletes) any sandbox idle >30 min — even
# if this daemon / the whole box dies. Idle-based, so it never touches an actively-running trial.
AUTO_STOP_MIN = 30
def _ensure_autostop(s):
    try:
        if (s.labels or {}).get("tmaxeval") == "1" and getattr(s, "auto_stop_interval", 0) in (0, None):
            s.set_autostop_interval(AUTO_STOP_MIN)
            return True
    except Exception:
        pass
    return False

while True:
    try:
        sbs = list(d.list())
        mine = [s for s in sbs if (s.labels or {}).get("tmaxeval") == "1"]
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            armed = sum(1 for r in ex.map(_ensure_autostop, mine) if r)
        victims = [s for s in sbs if _is_orphan(s)]
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            ok = sum(1 for r in ex.map(lambda s: (d.delete(s), True)[1] if True else False, victims) if r) if victims else 0
        print(f"{dt.datetime.now().strftime('%H:%M:%S')} total={len(sbs)} mine={len(mine)} "
              f"titan={sum(1 for s in sbs if (s.labels or {}).get('owner')=='titan_swe_r2e')} "
              f"autostop_armed={armed} orphans_deleted={ok}", flush=True)
    except Exception as e:
        print(f"daemon err: {e!r}"[:120], flush=True)
    time.sleep(300)
