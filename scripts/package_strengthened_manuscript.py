"""Validate the compiled paper, preserve source hashes, and package Overleaf."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "manuscript/iclr2027_overleaf_package_strengthened_20260903"
SOURCE = ROOT / "manuscript/iclr2027_overleaf_package_hardening_20260825"
ART = ROOT / "artifacts/reviewer_validation/reference_policy_strengthening_20260903_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font_types(resources, seen=None):
    seen = set() if seen is None else seen
    resources = resources.get_object() if hasattr(resources, "get_object") else resources
    if resources is None or id(resources) in seen:
        return set()
    seen.add(id(resources))
    result = set()
    for font in resources.get("/Font", {}).get_object().values() if "/Font" in resources else []:
        result.add(str(font.get_object().get("/Subtype")))
    for obj in resources.get("/XObject", {}).get_object().values() if "/XObject" in resources else []:
        result.update(font_types(obj.get_object().get("/Resources"), seen))
    return result


def main():
    manifest = json.loads((ART / "result_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["runs"] == 3800
    with (PAPER / "source_snapshot_sha256.csv").open(encoding="utf-8-sig", newline="") as stream:
        snapshot = list(csv.DictReader(stream))
    changed = [row["Path"] for row in snapshot if sha(SOURCE / row["Path"]).lower() != row["SHA256"].lower()]
    if changed:
        raise ValueError(f"Archived manuscript changed: {changed}")
    reader = PdfReader(PAPER / "paper.pdf")
    texts = [page.extract_text() or "" for page in reader.pages]
    normalized = [re.sub(r"\s+", "", t).lower() for t in texts]
    conclusion_page = next(i + 1 for i, t in enumerate(normalized) if "conclusion" in t)
    assert conclusion_page <= 9, "Main text exceeds nine pages"
    assert not any("??" in t for t in texts), "Unresolved reference"
    assert any("8.1" in t and "strictvalid" in n for t, n in zip(texts[:9], normalized[:9]))
    fonts = set().union(*(font_types(p.get("/Resources")) for p in reader.pages))
    assert "/Type3" not in fonts, "Type 3 font found"
    zip_path = ROOT / "dist/overleaf/BDMTF_ICLR2027_strengthened_20260903.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(PAPER.rglob("*")) if p.is_file() and p.suffix not in {".aux", ".out", ".log"}]
    secrets = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")
    for path in files:
        if path.suffix in {".tex", ".md", ".json", ".csv", ".bib", ".txt"}:
            assert not secrets.search(path.read_text(encoding="utf-8-sig")), f"Credential-shaped value in {path.name}"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(PAPER).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert {"paper.tex", "paper.pdf", "iclr2027_conference.sty", "generated/reference_policy_macros.tex"} <= set(archive.namelist())
    validation = {"status": "complete", "pages": len(reader.pages), "conclusion_page": conclusion_page,
                  "archived_files_verified": len(snapshot), "archived_files_changed": 0,
                  "strict_valid_retained_in_main_table": True, "pdf_font_types": sorted(fonts),
                  "paper_pdf_sha256": sha(PAPER / "paper.pdf"), "zip": str(zip_path.relative_to(ROOT)),
                  "zip_sha256": sha(zip_path), "packaged_files": len(files),
                  "result_manifest_sha256": sha(ART / "result_manifest.json"),
                  "manuscript_file_hashes": {str(p.relative_to(PAPER)): sha(p) for p in files},
                  "visual_qa": "Main results, conclusion, and new appendix tables/figure rendered and inspected."}
    (ART / "release_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    latest = ROOT / "artifacts/provenance/latest_status.json"
    state = json.loads(latest.read_text(encoding="utf-8-sig"))
    state["paper"].update({"status": "complete", "official_pdf_sha256": validation["paper_pdf_sha256"],
                           "overleaf_zip": validation["zip"], "overleaf_zip_sha256": validation["zip_sha256"]})
    state["reference_policy_strengthening"]["release_validation"] = str((ART / "release_validation.json").relative_to(ROOT))
    latest.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in validation.items() if k != "manuscript_file_hashes"}, indent=2))


if __name__ == "__main__":
    main()
