"""Validate and package the September 8 ICLR review-fixed manuscript."""

from __future__ import annotations

import hashlib
import csv
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "manuscript" / "iclr2027_overleaf_package_review_fixed_20260908"
PREVIOUS = ROOT / "manuscript" / "iclr2027_overleaf_package_integrated_20260907"
QREF = ROOT / "artifacts" / "reviewer_validation" / "qref_decomposition_20260908"
REVIEW = ROOT / "artifacts" / "reviewer_validation" / "review_20260908"
ZIP_PATH = ROOT / "dist" / "overleaf" / "BDMTF_ICLR2027_review_fixed_20260908.zip"
VALIDATION = REVIEW / "paper_release_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_source_snapshot() -> None:
    snapshot = PAPER / "source_snapshot_sha256.csv"
    excluded = {".aux", ".out", ".log"}
    files = [
        path
        for path in sorted(PAPER.rglob("*"))
        if path.is_file()
        and path != snapshot
        and path.suffix.lower() not in excluded
        and "rendered_review" not in path.relative_to(PAPER).parts
    ]
    with snapshot.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Path", "SHA256"])
        for path in files:
            writer.writerow(
                [path.relative_to(PAPER).as_posix(), sha256(path)]
            )


def main() -> None:
    qref = json.loads((QREF / "manifest.json").read_text(encoding="utf-8"))
    assert qref["status"] == "complete"
    assert qref["observed_runs"] == qref["expected_runs"] == 4000

    sparse = (REVIEW / "lemmy_sparse_path_baselines.csv").read_text(encoding="utf-8")
    assert "Always empty" in sparse and "0.029" in sparse

    aux = (PAPER / "paper.aux").read_text(encoding="utf-8", errors="replace")
    discussion = re.search(r"\\newlabel\{sec:discussion-identification\}\{\{6\}\{(\d+)\}", aux)
    assert discussion and int(discussion.group(1)) <= 9
    labels = re.findall(r"\\newlabel\{([^}]+)\}", aux)
    assert not [name for name, count in Counter(labels).items() if count > 1]

    reader = PdfReader(PAPER / "paper.pdf")
    assert len(reader.pages) == 33
    fonts = set().union(*(font_types(page.get("/Resources")) for page in reader.pages))
    assert "/Type3" not in fonts

    abstract = (PAPER / "sections" / "01_ABSTRACT.tex").read_text(encoding="utf-8")
    abstract_words = len(
        re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", re.sub(r"\\[A-Za-z]+", " ", abstract))
    )
    assert 180 <= abstract_words <= 220

    manuscript_text = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in sorted(PAPER.rglob("*.tex"))
    )
    assert "improves held-out Lemmy event-path fidelity" not in manuscript_text
    assert "catalog--coordinate decomposition" in manuscript_text.lower()
    assert "always-empty" in manuscript_text
    assert "??" not in manuscript_text

    write_source_snapshot()

    excluded = {".aux", ".out", ".log"}
    files = [
        path
        for path in sorted(PAPER.rglob("*"))
        if path.is_file()
        and path.suffix.lower() not in excluded
        and "rendered_review" not in path.relative_to(PAPER).parts
    ]
    secret_pattern = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")
    for path in files:
        if path.suffix.lower() in {".tex", ".md", ".json", ".csv", ".bib", ".txt"}:
            assert not secret_pattern.search(path.read_text(encoding="utf-8-sig")), f"credential in {path}"

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
            "generated/qref_decomposition_macros.tex",
            "generated/table_qref_decomposition.tex",
            "sections/22_APPENDIX_MECHANISM_PHASE.tex",
            "EXPERIMENT_REGISTRY.md",
        }
        assert required <= set(archive.namelist())

    previous_hash = sha256(PREVIOUS / "paper.pdf")
    assert previous_hash == "0a33cb2b29170d1e74efc9868bd025dc9125decd5d28a2e87a5fb1664bb79c2a"
    result = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pages": len(reader.pages),
        "main_text_discussion_page": int(discussion.group(1)),
        "abstract_words": abstract_words,
        "type3_fonts": False,
        "pdf_font_types": sorted(fonts),
        "qref_decomposition_runs": qref["observed_runs"],
        "previous_pdf_sha256": previous_hash,
        "paper_pdf": str((PAPER / "paper.pdf").relative_to(ROOT)).replace("\\", "/"),
        "paper_pdf_sha256": sha256(PAPER / "paper.pdf"),
        "overleaf_zip": str(ZIP_PATH.relative_to(ROOT)).replace("\\", "/"),
        "overleaf_zip_sha256": sha256(ZIP_PATH),
        "packaged_files": len(files),
        "visual_qa_pages": [1, 9, 27, 31],
    }
    VALIDATION.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
