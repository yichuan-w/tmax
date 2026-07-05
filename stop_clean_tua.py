#!/usr/bin/env python
"""Stop the leaking TUA run and drain ALL Daytona sandboxes.
TUA trials build+start sandboxes fine (STARTED) but fail afterward and harbor's teardown
can't remove them ("Failed to remove sandbox") -> orphans leak ~15/min -> quota saturates
-> new builds starve. Kill the run first (done in the wrapper shell), then loop-delete
until the account is near-empty (delete is eventually-consistent, so repeat)."""
import os, time, concurrent.futures as cf
from collections import Counter
from daytona import Daytona, DaytonaConfig
import daytona_api_client.configuration as _c
_orig = _c.Configuration.__init__
def _patched(self, *a, **k):
    _orig(self, *a, **k); self.proxy = os.environ.get("https_proxy")
_c.Configuration.__init__ = _patched

d = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))

def _del(s):
    try: d.delete(s); return True
    except Exception:
        try: d.delete(s); return True
        except Exception: return False

for rnd in range(1, 9):
    sbs = list(d.list())
    print(f"round {rnd}: total={len(sbs)} states={dict(Counter(str(getattr(s,'state',None)) for s in sbs))}", flush=True)
    if len(sbs) <= 3:
        print("account drained.", flush=True); break
    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        ok = sum(1 for r in ex.map(_del, sbs) if r)
    print(f"  round {rnd} deleted ~{ok}/{len(sbs)}", flush=True)
    time.sleep(15)
sbs = list(d.list())
print(f"FINAL total={len(sbs)} states={dict(Counter(str(getattr(s,'state',None)) for s in sbs))}", flush=True)
