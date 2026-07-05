#!/usr/bin/env python
"""Wrapper to run `harbor` with Daytona ephemeral sandboxes forced on (this Daytona
region only permits ephemeral sandboxes; harbor doesn't set the flag). AsyncDaytona
(aiohttp) picks up https_proxy from the env, so no proxy patch needed here.
Usage: python eval_harbor.py run --dataset ... (same args as the harbor CLI).
"""
import os
import sys
from daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams

# --- S3 build-context proxy fix (needed for TB-Pro / any Dockerfile-built task) ---
# TB-Pro/TUA tasks have no prebuilt docker_image, so harbor calls Image.from_dockerfile
# and the daytona SDK uploads the build context to S3 via obstore's Rust S3Store.
# That Rust reqwest client ignores https_proxy, AND the SDK wraps construction in
# isolated_env() which CLEARS all env vars first -> direct S3 egress -> blocked here
# -> "Failed to create sandbox: Generic S3 error ... context.tar" -> "Sandbox not found".
# Fix: replace S3Store in both object_storage modules with a factory that injects
# client_options.proxy_url (an EXPLICIT arg, so isolated_env clearing env is irrelevant).
_PROXY = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or "http://fwdproxy:8080"
try:
    from obstore.store import S3Store as _RealS3Store
    import daytona._async.object_storage as _aos
    import daytona._sync.object_storage as _sos

    def _proxied_s3store(*args, **kwargs):
        co = dict(kwargs.get("client_options") or {})
        co.setdefault("proxy_url", _PROXY)
        kwargs["client_options"] = co
        return _RealS3Store(*args, **kwargs)

    _aos.S3Store = _proxied_s3store
    _sos.S3Store = _proxied_s3store
except Exception as _e:  # pragma: no cover
    print(f"[eval_harbor] S3 proxy patch skipped: {_e!r}", file=sys.stderr)

for _cls in (CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams):
    if "ephemeral" in _cls.model_fields:
        _cls.model_fields["ephemeral"].default = True
        try:
            _cls.model_rebuild(force=True)
        except Exception:
            pass
    # belt-and-suspenders: force ephemeral on every construction + clamp resources
    # to this Daytona account's per-sandbox caps (CPU<=4, ephemeral disk<=10GB, mem<=8GB).
    # TUA-Bench tasks request cpus=6 / storage=30GB which Daytona rejects
    # ("CPU request 6 exceeds maximum allowed per sandbox (4)").
    _orig = _cls.__init__
    def _mk(orig):
        def _init(self, **kw):
            kw.setdefault("ephemeral", True)
            # Tag every sandbox we create so an orphan-cleanup daemon can delete ONLY
            # ours and NEVER touch other workloads sharing this Daytona key (there's an
            # active torchtitan `owner=titan_swe_r2e` RL run on the same account!).
            lbl = dict(kw.get("labels") or {})
            lbl["tmaxeval"] = "1"
            kw["labels"] = lbl
            res = kw.get("resources")
            if res is not None:
                try:
                    if getattr(res, "cpu", None) and res.cpu > 4: res.cpu = 4
                    if getattr(res, "memory", None) and res.memory > 8: res.memory = 8
                    if getattr(res, "disk", None) and res.disk > 10: res.disk = 10
                except Exception:
                    pass
            orig(self, **kw)
        return _init
    _cls.__init__ = _mk(_orig)

from harbor.cli.main import app

sys.argv = ["harbor"] + sys.argv[1:]
app()
