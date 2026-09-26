from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESKTOP = Path.home() / "Desktop"
SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|CLIENT_SECRET\s*=\s*['\"][^'\"]{8,}|REDDIT_CLIENT_SECRET\s*=\s*['\"][^'\"]{8,})"
)


def ok(item: str, detail: str = "") -> Dict[str, Any]:
    return {"status": "ok", "item": item, "detail": detail}


def warn(item: str, detail: str = "") -> Dict[str, Any]:
    return {"status": "warn", "item": item, "detail": detail}


def fail(item: str, detail: str = "") -> Dict[str, Any]:
    return {"status": "fail", "item": item, "detail": detail}


def check_file(path: Path, label: str) -> Dict[str, Any]:
    return ok(label, str(path)) if path.exists() else fail(label, f"missing: {path}")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_core_files() -> List[Dict[str, Any]]:
    checks = [
        check_file(ROOT / "configs" / "paper_reproduction.json", "paper config"),
        check_file(ROOT / "data" / "social_paper" / "manifest.json", "paper dataset manifest"),
        check_file(ROOT / "data" / "social_paper" / "calibration_constants.json", "community calibration constants"),
        check_file(ROOT / "data" / "source_archives_inventory.json", "source archive inventory"),
        check_file(ROOT / "src" / "bdmtf" / "simulator.py", "BDMTF simulator"),
        check_file(ROOT / "src" / "bdmtf" / "oasis_bridge.py", "OASIS bridge"),
        check_file(ROOT / "scripts" / "check_oasis_integration.py", "OASIS integration check"),
        check_file(ROOT / "scripts" / "check_run_outputs.py", "run output completeness check"),
        check_file(ROOT / "scripts" / "reproduce_paper_resumable.py", "resumable paper reproduction entrypoint"),
        check_file(ROOT / "scripts" / "check_api_environment.py", "API environment check"),
        check_file(ROOT / "scripts" / "vendor_upstream_oasis.py", "upstream OASIS vendoring script"),
        check_file(ROOT / "examples" / "oasis_shim_post.json", "OASIS shim example post"),
        check_file(ROOT / "scripts" / "reproduce_paper.py", "paper reproduction entrypoint"),
        check_file(ROOT / "scripts" / "run_ablation.py", "ablation entrypoint"),
        check_file(ROOT / "scripts" / "run_parameter_sensitivity.py", "sensitivity entrypoint"),
        check_file(ROOT / "configs" / "api_intent_generation.json", "API intent generation config"),
        check_file(ROOT / ".env.api.example", "API env example"),
    ]
    return checks


def audit_dataset() -> List[Dict[str, Any]]:
    manifest_path = ROOT / "data" / "social_paper" / "manifest.json"
    if not manifest_path.exists():
        return [fail("dataset", "manifest missing")]
    manifest = load_json(manifest_path)
    checks: List[Dict[str, Any]] = []
    communities = manifest.get("communities", {})
    if isinstance(communities, list):
        return [fail("dataset manifest schema", "expected community mapping, found list")]
    for community, info in communities.items():
        base = ROOT / "data" / "social_paper" / f"{community}_data"
        if not base.exists():
            checks.append(fail(f"{community} directory", str(base)))
            continue
        for name in info.get("files", []):
            path = base / name
            checks.append(check_file(path, f"{community}/{name}"))
        posts_path = base / f"posts_features_{community}.csv"
        if posts_path.exists():
            try:
                rows = len(pd.read_csv(posts_path, usecols=["post_id"]))
                expected = int(info.get("selected_posts", 100))
                if rows == expected:
                    checks.append(ok(f"{community} selected posts", str(rows)))
                else:
                    checks.append(fail(f"{community} selected posts", f"{rows} != {expected}"))
            except Exception as exc:  # pragma: no cover - audit diagnostics
                checks.append(fail(f"{community} posts csv readable", f"{type(exc).__name__}: {exc}"))
        comments_path = base / f"comments_data_{community}.csv"
        if comments_path.exists():
            try:
                rows = len(pd.read_csv(comments_path, usecols=["post_id"]))
                expected = int(info.get("comment_rows", 0))
                if rows == expected:
                    checks.append(ok(f"{community} comment rows", str(rows)))
                else:
                    checks.append(warn(f"{community} comment rows", f"{rows} != manifest {expected}"))
            except Exception as exc:  # pragma: no cover
                checks.append(fail(f"{community} comments csv readable", f"{type(exc).__name__}: {exc}"))
    return checks


def audit_calibration() -> List[Dict[str, Any]]:
    config_path = ROOT / "configs" / "paper_reproduction.json"
    constants_path = ROOT / "data" / "social_paper" / "calibration_constants.json"
    if not config_path.exists() or not constants_path.exists():
        return [fail("calibration", "config or constants missing")]
    config = load_json(config_path)
    constants = load_json(constants_path)
    config_cal = config.get("community_calibration", {})
    checks: List[Dict[str, Any]] = []
    for community, raw in constants.get("communities", {}).items():
        configured = config_cal.get(community)
        if not configured:
            checks.append(fail(f"{community} calibration", "missing from config"))
            continue
        for key in ("threshold_base", "early_engagement_median"):
            a = float(raw[key])
            b = float(configured[key])
            if abs(a - b) < 1e-9:
                checks.append(ok(f"{community} {key}", str(a)))
            else:
                checks.append(fail(f"{community} {key}", f"{b} != {a}"))
    return checks


def audit_legacy_and_sources(desktop: Path) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    legacy = ROOT / "legacy" / "social_pipeline" / "sanitized_scripts"
    required_legacy = [
        "reddit_data_collector.py",
        "1-llm_feature_extractor_api_safe.py",
        "2-oasis_calibration_calculator.py",
        "3-oasis_profile_generator.py",
        "0extract_empirical_cascades.py",
        "run_oasis_simulation.py",
        "_oasis_task_executor.py",
    ]
    for name in required_legacy:
        checks.append(check_file(legacy / name, f"legacy {name}"))
    if legacy.exists():
        leaked = []
        for path in legacy.glob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if SECRET_RE.search(text):
                leaked.append(path.name)
        if leaked:
            checks.append(fail("legacy secret scan", ", ".join(leaked)))
        else:
            checks.append(ok("legacy secret scan", "no raw API keys or Reddit secrets found"))

    for name in ["social", "social.zip", "reddit数据1.zip", "reddit数据2.zip"]:
        path = desktop / name
        if path.exists():
            detail = str(path)
            if path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(path) as zf:
                        detail += f" entries={len(zf.infolist())}"
                except Exception as exc:  # pragma: no cover
                    detail += f" zip-error={type(exc).__name__}: {exc}"
            checks.append(ok(f"source {name}", detail))
        else:
            checks.append(warn(f"source {name}", f"not found at {path}"))
    return checks


def audit_oasis() -> List[Dict[str, Any]]:
    oasis_installed = importlib.util.find_spec("oasis") is not None
    camel_installed = importlib.util.find_spec("camel") is not None
    upstream = ROOT / "vendor" / "oasis" / "upstream"
    upstream_manifest = ROOT / "vendor" / "oasis" / "upstream_manifest.json"
    upstream_files = [path for path in upstream.rglob("*") if path.is_file()] if upstream.exists() else []
    upstream_populated = len([path for path in upstream_files if path.name != "README.md"]) > 0
    checks = [
        ok("OASIS Python package", "installed") if oasis_installed else warn("OASIS Python package", "not installed"),
        ok("CAMEL Python package", "installed") if camel_installed else warn("CAMEL Python package", "not installed"),
        check_file(ROOT / "requirements-oasis.txt", "OASIS dependency file"),
        check_file(ROOT / "requirements-legacy.txt", "legacy dependency file"),
        check_file(ROOT / "vendor" / "oasis" / "README.md", "OASIS vendoring note"),
        check_file(ROOT / "OASIS_UPSTREAM_STATUS.md", "OASIS upstream status note"),
        ok("upstream OASIS source", str(upstream))
        if upstream_populated
        else warn("upstream OASIS source", f"not populated; use scripts/vendor_upstream_oasis.py to fill {upstream}"),
        ok("upstream OASIS manifest", str(upstream_manifest))
        if upstream_manifest.exists()
        else warn("upstream OASIS manifest", "not present until upstream source is vendored"),
        check_file(ROOT / "vendor" / "oasis" / "shim" / "README.md", "OASIS shim note"),
        check_file(ROOT / "vendor" / "oasis" / "shim" / "oasis" / "__init__.py", "OASIS local shim"),
        check_file(ROOT / "vendor" / "oasis" / "shim" / "camel" / "models.py", "CAMEL local shim"),
    ]
    return checks


def run_audit(desktop: Path) -> List[Dict[str, Any]]:
    return (
        audit_core_files()
        + audit_dataset()
        + audit_calibration()
        + audit_legacy_and_sources(desktop)
        + audit_oasis()
    )


def print_report(checks: List[Dict[str, Any]]) -> None:
    for item in checks:
        print(f"[{item['status'].upper()}] {item['item']}: {item['detail']}")
    failures = [item for item in checks if item["status"] == "fail"]
    warnings = [item for item in checks if item["status"] == "warn"]
    print(f"\nSummary: {len(failures)} failures, {len(warnings)} warnings, {len(checks)} checks")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit all inputs needed for the BDMTF paper reproduction.")
    parser.add_argument("--desktop", default=str(DEFAULT_DESKTOP), help="Desktop/source folder containing raw archives.")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    checks = run_audit(Path(args.desktop))
    if args.json:
        print(json.dumps(checks, indent=2, ensure_ascii=False))
    else:
        print_report(checks)
    return 1 if any(item["status"] == "fail" for item in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
