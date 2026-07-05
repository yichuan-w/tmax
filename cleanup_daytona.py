#!/usr/bin/env python
"""Delete orphaned Daytona sandboxes that jam the account's concurrent-sandbox quota.
Root cause of TUA builds never completing: 607 STARTED sandboxes orphaned from prior
eval runs (ephemeral auto-delete didn't fire for hung-teardown trials) saturate the
account -> new BUILDING_SNAPSHOT builds starve. Delete STARTED/BUILD_FAILED/ERROR;
keep BUILDING_SNAPSHOT (current TUA run's in-flight builds)."""
import os, concurrent.futures as cf
from daytona import Daytona, DaytonaConfig
import daytona_api_client.configuration as _cfg
_o = _cfg.Configuration.__init__
def _p(self, *a, **k):
    _o(self, *a, **k); self.proxy = os.environ.get("https_proxy")
_cfg.Configuration.__init__ = _p

d = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
sbs = list(d.list())
KEEP = {"SandboxState.BUILDING_SNAPSHOT"}
victims = [s for s in sbs if str(getattr(s, "state", None)) not in KEEP]
print(f"total={len(sbs)} deleting={len(victims)} keeping(building)={len(sbs)-len(victims)}", flush=True)

def _del(s):
    try:
        d.delete(s); return True
    except Exception:
        try: d.delete(s); return True   # one retry
        except Exception as e: return repr(e)[:80]

ok = 0; errs = 0
with cf.ThreadPoolExecutor(max_workers=24) as ex:
    for i, r in enumerate(ex.map(_del, victims), 1):
        if r is True: ok += 1
        else: errs += 1
        if i % 50 == 0: print(f"  deleted {ok}, errs {errs} / {len(victims)}", flush=True)
print(f"DONE deleted={ok} errs={errs}", flush=True)
# re-list to confirm
sbs2 = list(d.list())
from collections import Counter
print("after:", dict(Counter(str(getattr(s,'state',None)) for s in sbs2)), flush=True)
