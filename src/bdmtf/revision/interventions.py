from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bdmtf.revision.external_data import read_table
from bdmtf.revision.provenance import sha256_file, write_json


REQUIRED_INTERVENTION_COLUMNS = (
    "unit_id",
    "domain",
    "treatment",
    "event_time",
    "pre_outcome",
    "post_outcome",
)


def intervention_readiness(frame: pd.DataFrame) -> dict[str, Any]:
    missing = [name for name in REQUIRED_INTERVENTION_COLUMNS if name not in frame]
    if missing:
        return {"status": "invalid", "claim_allowed": False, "missing_columns": missing}
    treatment = pd.to_numeric(frame["treatment"], errors="coerce")
    event_time = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    issues = []
    if treatment.isna().any() or not set(treatment.dropna().astype(int).unique()).issubset({0, 1}):
        issues.append("treatment must be binary")
    if treatment.nunique() < 2:
        issues.append("both treated and control units are required")
    if event_time.isna().any():
        issues.append("event_time contains invalid timestamps")
    if frame["unit_id"].astype(str).duplicated().any():
        issues.append("unit_id must be unique")
    for name in ("pre_outcome", "post_outcome"):
        if pd.to_numeric(frame[name], errors="coerce").isna().any():
            issues.append(f"{name} must be numeric and complete")
    return {
        "status": "ready" if not issues else "invalid",
        "claim_allowed": not issues,
        "issues": issues,
        "n_units": int(len(frame)),
        "n_treated": int((treatment == 1).sum()),
        "n_control": int((treatment == 0).sum()),
        "n_domains": int(frame["domain"].nunique()),
    }


def _standardized_mean_difference(treated: np.ndarray, control: np.ndarray) -> float:
    denominator = np.sqrt((np.var(treated, ddof=1) + np.var(control, ddof=1)) / 2.0)
    return 0.0 if denominator < 1e-12 else float((np.mean(treated) - np.mean(control)) / denominator)


def estimate_matched_difference_in_differences(
    frame: pd.DataFrame,
    covariates: Iterable[str],
    bootstrap_samples: int = 1000,
    seed: int = 30371,
) -> tuple[dict[str, Any], pd.DataFrame]:
    readiness = intervention_readiness(frame)
    if not readiness["claim_allowed"]:
        raise ValueError(f"Intervention data are not analysis-ready: {readiness}")
    covariates = [str(name) for name in covariates]
    missing = [name for name in covariates if name not in frame]
    if missing:
        raise ValueError(f"Missing intervention covariates: {', '.join(missing)}")

    working = frame.copy()
    treatment = working["treatment"].astype(int).to_numpy()
    raw_covariates = working[covariates].apply(pd.to_numeric, errors="coerce")
    if raw_covariates.isna().any().any():
        raise ValueError("Intervention covariates must be numeric and complete")
    design = StandardScaler().fit_transform(raw_covariates.to_numpy(dtype=float))
    propensity = LogisticRegression(max_iter=2000, random_state=seed).fit(design, treatment).predict_proba(design)[:, 1]
    working["propensity_score"] = propensity
    treated_indices = np.flatnonzero(treatment == 1)
    control_indices = np.flatnonzero(treatment == 0)
    matched_controls = np.asarray(
        [control_indices[np.argmin(np.abs(propensity[control_indices] - propensity[index]))] for index in treated_indices],
        dtype=int,
    )
    changes = pd.to_numeric(working["post_outcome"]) - pd.to_numeric(working["pre_outcome"])
    pair_effects = changes.iloc[treated_indices].to_numpy(dtype=float) - changes.iloc[matched_controls].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [rng.choice(pair_effects, size=len(pair_effects), replace=True).mean() for _ in range(bootstrap_samples)],
        dtype=float,
    )

    pairs = pd.DataFrame(
        {
            "treated_unit_id": working.iloc[treated_indices]["unit_id"].astype(str).to_numpy(),
            "control_unit_id": working.iloc[matched_controls]["unit_id"].astype(str).to_numpy(),
            "treated_domain": working.iloc[treated_indices]["domain"].astype(str).to_numpy(),
            "propensity_distance": np.abs(propensity[treated_indices] - propensity[matched_controls]),
            "pair_effect": pair_effects,
        }
    )
    balance = {}
    for index, name in enumerate(covariates):
        balance[name] = {
            "smd_before": _standardized_mean_difference(design[treated_indices, index], design[control_indices, index]),
            "smd_after": _standardized_mean_difference(design[treated_indices, index], design[matched_controls, index]),
        }
    summary = {
        **readiness,
        "estimator": "nearest-propensity matched difference-in-differences with replacement",
        "estimand": "average treatment effect on the treated",
        "effect": float(pair_effects.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_samples": int(bootstrap_samples),
        "covariates": covariates,
        "balance": balance,
        "causal_caveat": "Validity still requires a defensible intervention time, overlap, no unmeasured time-varying confounding, and parallel trends.",
    }
    return summary, pairs


def analyze_intervention_file(root: Path, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    intervention = config.get("intervention", {})
    configured = str(intervention.get("path", "")).strip()
    if not configured:
        result = {
            "status": "needs_data",
            "claim_allowed": False,
            "required_columns": list(REQUIRED_INTERVENTION_COLUMNS),
            "reason": "No real intervention dataset is configured.",
            "note": "Removed comment text without removal timestamps is not a valid intervention event.",
        }
        write_json(output_dir / "intervention_audit.json", result)
        return result
    path = Path(configured)
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        result = {
            "status": "needs_data",
            "claim_allowed": False,
            "path": str(path),
            "reason": "Configured intervention file does not exist.",
        }
        write_json(output_dir / "intervention_audit.json", result)
        return result

    frame = read_table(path)
    readiness = intervention_readiness(frame)
    readiness.update({"path": str(path), "sha256": sha256_file(path)})
    write_json(output_dir / "intervention_audit.json", readiness)
    if not readiness["claim_allowed"]:
        return readiness
    summary, pairs = estimate_matched_difference_in_differences(
        frame,
        intervention.get("covariates", []),
        int(intervention.get("bootstrap_samples", 1000)),
        int(intervention.get("seed", 30371)),
    )
    summary.update({"path": str(path), "sha256": sha256_file(path)})
    pairs.to_csv(output_dir / "matched_pairs.csv", index=False)
    write_json(output_dir / "intervention_effect.json", summary)
    return summary
