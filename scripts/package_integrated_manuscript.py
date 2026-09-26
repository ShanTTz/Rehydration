"""Validate and package the cross-task integrated ICLR manuscript."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "manuscript/iclr2027_overleaf_package_integrated_20260907"
PREVIOUS = ROOT / "manuscript/iclr2027_overleaf_package_strengthened_20260903"
IMPORTED = ROOT / "artifacts/reviewer_validation/mechanism_storyline_20260901"
PROVENANCE = ROOT / "artifacts/provenance/cross_task_mechanism_v6"
VALIDATION_DIR = ROOT / "artifacts/reviewer_validation/integrated_paper_20260907"
ZIP_PATH = ROOT / "dist/overleaf/BDMTF_ICLR2027_integrated_20260907.zip"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font_types(resources, seen=None):
    seen = set() if seen is None else seen
    resources = resources.get_object() if hasattr(resources, "get_object") else resources
    if resources is None or id(resources) in seen:
        return set()
    seen.add(id(resources))
    result = set()
    fonts = resources.get("/Font", {}).get_object().values() if "/Font" in resources else []
    for font in fonts:
        result.add(str(font.get_object().get("/Subtype")))
    objects = resources.get("/XObject", {}).get_object().values() if "/XObject" in resources else []
    for obj in objects:
        result.update(font_types(obj.get_object().get("/Resources"), seen))
    return result


def main() -> None:
    qref = json.loads((IMPORTED / "qref_factorial_replication/manifest.json").read_text(encoding="utf-8"))
    core = json.loads((IMPORTED / "core_phase_precision_extension/manifest.json").read_text(encoding="utf-8"))
    reference = json.loads(
        (ROOT / "artifacts/reviewer_validation/reference_policy_strengthening_20260903_v2/result_manifest.json")
        .read_text(encoding="utf-8")
    )
    assert qref["status"] == "complete" and qref["execution"]["merged_runs"] == 4800
    assert qref["diagnostics"]["coordinate_linked_support_families"] == 3
    assert core["status"] == "complete" and core["diagnostics"]["composite_support_cells"] == 16
    assert reference["status"] == "complete" and reference["runs"] == 3800

    reader = PdfReader(PAPER / "paper.pdf")
    texts = [page.extract_text() or "" for page in reader.pages]
    normalized = [re.sub(r"\s+", "", text).lower() for text in texts]
    conclusion_page = next(i + 1 for i, text in enumerate(normalized) if "conclusion" in text)
    assert conclusion_page <= 9, "Scientific main text exceeds nine pages"
    assert "aiusestatement" in normalized[9], "AI Use Statement does not begin after the main text"
    assert not any("??" in text for text in texts), "Unresolved reference found"

    main_text = "".join(normalized[:9])
    for token in ("1.214", "24/25", "0.044", "0.070", "16/16"):
        assert token in main_text, f"Expected integrated result missing from main text: {token}"

    abstract = (PAPER / "sections/01_ABSTRACT.tex").read_text(encoding="utf-8")
    abstract_words = len(re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", re.sub(r"\\[A-Za-z]+", " ", abstract)))
    assert 180 <= abstract_words <= 220

    fonts = set().union(*(font_types(page.get("/Resources")) for page in reader.pages))
    assert "/Type3" not in fonts, "Type 3 font found"

    excluded = {".aux", ".out", ".log"}
    files = [path for path in sorted(PAPER.rglob("*")) if path.is_file() and path.suffix not in excluded]
    secret_pattern = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")
    for path in files:
        if path.suffix.lower() in {".tex", ".md", ".json", ".csv", ".bib", ".txt"}:
            assert not secret_pattern.search(path.read_text(encoding="utf-8-sig")), f"Credential in {path}"

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(PAPER).as_posix())
    with zipfile.ZipFile(ZIP_PATH) as archive:
        assert archive.testzip() is None
        required = {
            "paper.tex",
            "paper.pdf",
            "iclr2027_conference.sty",
            "generated/qref_factorial_macros.tex",
            "generated/core_phase_confirmation.pdf",
            "sections/22_APPENDIX_MECHANISM_PHASE.tex",
        }
        assert required <= set(archive.namelist())

    source_zip = PROVENANCE / "mechanism_revision_source_v6_story_abstract_22p.zip"
    imported_files = [path for path in sorted(IMPORTED.rglob("*")) if path.is_file()]
    previous_pdf_sha = sha256(PREVIOUS / "paper.pdf")
    validation = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pages": len(reader.pages),
        "main_text_conclusion_page": conclusion_page,
        "abstract_words": abstract_words,
        "unresolved_references": 0,
        "type3_fonts": False,
        "pdf_font_types": sorted(fonts),
        "imported_mechanism_files": len(imported_files),
        "qref_factorial_runs": qref["execution"]["merged_runs"],
        "qref_support_families": qref["diagnostics"]["coordinate_linked_support_families"],
        "core_phase_composite_support": core["diagnostics"]["composite_support_cells"],
        "reference_policy_runs_retained": reference["runs"],
        "source_v6_zip_sha256": sha256(source_zip),
        "previous_pdf_sha256": previous_pdf_sha,
        "paper_pdf_sha256": sha256(PAPER / "paper.pdf"),
        "overleaf_zip": str(ZIP_PATH.relative_to(ROOT)),
        "overleaf_zip_sha256": sha256(ZIP_PATH),
        "packaged_files": len(files),
        "visual_qa": "Pages 1-10 and imported mechanism appendix pages 32-33 rendered and inspected.",
        "imported_file_hashes": {
            str(path.relative_to(IMPORTED)).replace("\\", "/"): sha256(path) for path in imported_files
        },
    }
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    validation_path = VALIDATION_DIR / "release_validation.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    latest_path = ROOT / "artifacts/provenance/latest_status.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8-sig"))
    old_paper = dict(latest["paper"])
    latest["generated_at"] = validation["generated_at"]
    latest["paper"].update(
        {
            "status": "complete",
            "official_tex": str((PAPER / "paper.tex").relative_to(ROOT)),
            "official_pdf": str((PAPER / "paper.pdf").relative_to(ROOT)),
            "official_pdf_sha256": validation["paper_pdf_sha256"],
            "overleaf_zip": validation["overleaf_zip"],
            "overleaf_zip_sha256": validation["overleaf_zip_sha256"],
            "preserved_strengthened_20260903": {
                "official_pdf": old_paper.get("official_pdf"),
                "official_pdf_sha256": old_paper.get("official_pdf_sha256"),
                "overleaf_zip": old_paper.get("overleaf_zip"),
                "overleaf_zip_sha256": old_paper.get("overleaf_zip_sha256"),
            },
        }
    )
    latest["cross_task_mechanism_integration"] = {
        "status": "complete",
        "source_task_id": "01a03916-ed0c-71e0-8a44-c1558aa4d743",
        "source_commit": "4632e4e",
        "source_v6_pdf_sha256": "5686ad12e021aee5aac19e410afa9fbc21cd437f1abb4e1127d348f5f9dbcced",
        "qref_factorial_runs": 4800,
        "qref_support_families": "3/3",
        "core_phase_support": "16/16",
        "validation": str(validation_path.relative_to(ROOT)),
    }
    latest["latest_one_sentence"] = (
        "已将另一任务的四单元机制、三族独立Q_ref与冻结核心相位证据整合进最新ICLR稿；"
        "正文9页，Lemmy路径与激活函数补强均保留。"
    )
    latest_path.write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({key: value for key, value in validation.items() if key != "imported_file_hashes"}, indent=2))


if __name__ == "__main__":
    main()
