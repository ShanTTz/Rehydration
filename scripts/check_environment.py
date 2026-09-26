from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


CORE_PACKAGES = {
    "numpy": "1.23",
    "pandas": "1.5",
}


def parse_version(value: str) -> Tuple[int, ...]:
    parts = []
    for token in value.replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                parts.append(int(digits))
            break
    return tuple(parts)


def check_package(name: str, minimum: str | None = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"name": name, "installed": False}
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostics only
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    version = getattr(module, "__version__", "unknown")
    result.update({"installed": True, "version": version})
    if minimum and version != "unknown":
        result["minimum"] = minimum
        result["meets_minimum"] = parse_version(version) >= parse_version(minimum)
    return result


def file_exists(path: Path) -> Dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "exists": path.exists()}


def load_manifest() -> Dict[str, Any]:
    manifest_path = ROOT / "data" / "social_paper" / "manifest.json"
    if not manifest_path.exists():
        return {"exists": False, "communities": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "communities": manifest.get("communities", []),
        "post_count_per_community": manifest.get("posts_per_community"),
    }


def api_status() -> Dict[str, Any]:
    key = os.environ.get("OPENAI_API_KEY", "")
    return {
        "OPENAI_API_KEY": "set" if key else "missing",
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", "missing"),
        "OPENAI_MODEL": os.environ.get("OPENAI_MODEL", "missing"),
    }


def build_report() -> Dict[str, Any]:
    core = [check_package(name, minimum) for name, minimum in CORE_PACKAGES.items()]
    required_files = [
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
        ROOT / "requirements-lock.txt",
        ROOT / "configs" / "paper_reproduction.json",
        ROOT / "data" / "social_paper" / "calibration_constants.json",
        ROOT / "scripts" / "reproduce_paper.py",
        ROOT / "src" / "bdmtf" / "simulator.py",
    ]
    optional = {
        "oasis": check_package("oasis"),
        "camel": check_package("camel"),
    }
    return {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "meets_minimum": sys.version_info >= (3, 10),
        },
        "core_packages": core,
        "optional_packages": optional,
        "required_files": [file_exists(path) for path in required_files],
        "paper_dataset": load_manifest(),
        "api_environment": api_status(),
    }


def report_ok(report: Dict[str, Any]) -> bool:
    if not report["python"]["meets_minimum"]:
        return False
    for package in report["core_packages"]:
        if not package.get("installed") or not package.get("meets_minimum", True):
            return False
    for required in report["required_files"]:
        if not required["exists"]:
            return False
    if not report["paper_dataset"]["exists"]:
        return False
    return True


def print_human(report: Dict[str, Any]) -> None:
    status = "OK" if report_ok(report) else "FAILED"
    print(f"BDMTF environment check: {status}")
    print(f"Python: {report['python']['version']}")

    print("\nCore packages:")
    for package in report["core_packages"]:
        if package.get("installed"):
            minimum = package.get("minimum", "-")
            print(f"  OK {package['name']} {package['version']} (minimum {minimum})")
        else:
            print(f"  MISSING {package['name']} - {package.get('error', '')}")

    print("\nRequired files:")
    for required in report["required_files"]:
        prefix = "OK" if required["exists"] else "MISSING"
        print(f"  {prefix} {required['path']}")

    dataset = report["paper_dataset"]
    if dataset["exists"]:
        communities = ", ".join(dataset["communities"])
        print(f"\nPaper dataset: OK ({communities})")
    else:
        print("\nPaper dataset: MISSING data/social_paper/manifest.json")

    print("\nOptional packages:")
    for name, package in report["optional_packages"].items():
        prefix = "OK" if package.get("installed") else "not installed"
        version = package.get("version", "")
        print(f"  {prefix} {name} {version}".rstrip())

    api = report["api_environment"]
    print("\nAPI environment:")
    print(f"  OPENAI_API_KEY: {api['OPENAI_API_KEY']}")
    print(f"  OPENAI_BASE_URL: {api['OPENAI_BASE_URL']}")
    print(f"  OPENAI_MODEL: {api['OPENAI_MODEL']}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check BDMTF reproduction environment.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_human(report)
    return 0 if report_ok(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
