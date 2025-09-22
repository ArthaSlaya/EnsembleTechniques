from __future__ import annotations
import argparse, os, json
from datetime import datetime, timezone
from src.aaa.exp.dataio import load_data

def main():
    ap = argparse.ArgumentParser(description="Diagnostic check for dataio module")
    ap.add_argument("--data", required=True, help="file/folder/glob path")
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()

    outdir = "artifacts/diagnostics/dataio"
    os.makedirs(outdir, exist_ok=True)

    df = load_data(args.data, limit=args.limit)

    summary = {
        "module": "dataio",
        "status": "pass" if len(df) > 0 and df.shape[1] > 0 else "fail",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"path_arg": args.data, "limit": args.limit},
        "resolved": {
            "rows_loaded": int(len(df)),
            "cols_loaded": int(df.shape[1]),
            "columns_sample": df.columns.tolist()[:10],
        },
        "conditions": {
            "rows_loaded_gt_0": len(df) > 0,
            "cols_loaded_gt_0": df.shape[1] > 0,
        },
        "artifacts": {
            "sample_csv": f"{outdir}/sample_head.csv",
            "summary_json": f"{outdir}/summary.json",
        },
    }

    df.head(20).to_csv(summary["artifacts"]["sample_csv"], index=False)
    with open(summary["artifacts"]["summary_json"], "w") as f:
        json.dump(summary, f, indent=2)

    # Append to global audit log
    with open("artifacts/diagnostics/audit_log.jsonl", "a") as audit_f:
        audit_f.write(json.dumps(summary) + "\n")

    print(f"[DataIO] {summary['status'].upper()}: {len(df)} rows, {df.shape[1]} cols")
    print(f"Artifacts → {outdir}/")

if __name__ == "__main__":
    main()


python -m src.aaa.diagnostics.dataio_check \
  --data data_stream/processed/date_2024-01-*/part-*.parquet \
  --limit 5000


#===============================================

mkdir -p src/aaa/exp
mkdir -p src/aaa/diagnostics
mkdir -p artifacts/diagnostics/features

# package markers (ok if already present)
touch src/__init__.py
touch src/aaa/__init__.py
touch src/aaa/exp/__init__.py
touch src/aaa/diagnostics/__init__.py

# module + diagnostic files
touch src/aaa/exp/features.py
touch src/aaa/diagnostics/features_check.py

from __future__ import annotations
import hashlib
from typing import Dict, List, Optional
import pandas as pd
import yaml

"""
YAML structure supported:

id: fs_v1
i_columns:
  - device_id
  - date

feature_groups:
  SC:
    - SC_session_count
    - SC_expanding_mean
  SS:
    - SS_ShortSessionCounts
    - SS_ShortSessionAnomaly

feature_all:
  - SC_session_count
  - SS_ShortSessionCounts
"""

def load_feature_spec(path: str) -> Dict:
    with open(path, "r") as f:
        spec = yaml.safe_load(f) or {}
    if "id" not in spec:
        raise ValueError("Feature spec must include an 'id' field")
    spec.setdefault("i_columns", [])
    spec.setdefault("feature_groups", {})
    spec.setdefault("feature_all", [])
    return spec

def resolve_feature_columns(
    spec: Dict,
    mode: str = "all",
    groups: Optional[List[str]] = None,
    extra: Optional[List[str]] = None,
) -> List[str]:
    """
    mode:
      - 'all'    -> use feature_all if present, else union of all groups
      - 'groups' -> use only specified groups (list in `groups`)
      - 'list'   -> use explicit list in `extra`
    """
    groups = groups or []
    extra = extra or []
    cols: List[str] = []

    if mode == "all":
        if spec.get("feature_all"):
            cols = list(spec["feature_all"])  # preserve YAML order
        else:
            for _, glist in (spec.get("feature_groups") or {}).items():
                cols.extend(glist)
    elif mode == "groups":
        if not groups:
            raise ValueError("mode='groups' requires a non-empty groups list")
        fg = spec.get("feature_groups") or {}
        for g in groups:
            if g not in fg:
                raise KeyError(f"Group '{g}' not found in feature_groups")
            cols.extend(fg[g])
    elif mode == "list":
        if not extra:
            raise ValueError("mode='list' requires explicit 'extra' columns")
        cols = list(extra)
    else:
        raise ValueError(f"Unknown feature resolution mode: {mode}")

    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped

def select_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:8])
        more = "..." if len(missing) > 8 else ""
        raise KeyError(f"Missing columns in data: {preview}{more}")

    sel = df[columns].copy()

    # Coerce object→numeric where possible
    for c in columns:
        if sel[c].dtype == object:
            sel[c] = pd.to_numeric(sel[c], errors="coerce")

    # Simple numeric imputation for NaNs
    sel = sel.fillna(sel.median(numeric_only=True))

    # Drop any leftover non-numeric columns to keep models safe
    non_numeric = sel.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        sel = sel.drop(columns=non_numeric)

    return sel

def feature_manifest(columns: List[str]) -> Dict:
    ordered = list(columns)
    joined = "|".join(ordered)
    h = hashlib.sha1(joined.encode()).hexdigest()
    return {"n_features": len(ordered), "feature_hash": h, "columns": ordered}
    
#................................
from __future__ import annotations
import argparse, os, json
from datetime import datetime, timezone

import pandas as pd

from src.aaa.exp.dataio import load_data
from src.aaa.exp.features import (
    load_feature_spec,
    resolve_feature_columns,
    select_features,
    feature_manifest,
)

def main():
    ap = argparse.ArgumentParser(description="Diagnostic check for features module")
    ap.add_argument("--data", required=True, help="file/folder/glob path")
    ap.add_argument("--features", required=True, help="features YAML")
    ap.add_argument("--feature-mode", choices=["all","groups","list"], default="all")
    ap.add_argument("--feature-groups", default=None, help="comma list when mode=groups (e.g., SC,SS)")
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()

    outdir = "artifacts/diagnostics/features"
    os.makedirs(outdir, exist_ok=True)

    # Load YAML
    spec = load_feature_spec(args.features)
    groups = [g.strip() for g in args.feature_groups.split(",")] if args.feature_groups else None
    cols = resolve_feature_columns(spec, mode=args.feature_mode, groups=groups)

    # Load data & select features
    raw = load_data(args.data, limit=args.limit)
    X = select_features(raw, cols)
    manifest = feature_manifest(list(X.columns))

    # Data quality checks
    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    nan_count = int(X.isna().sum().sum())  # after impute should be 0
    status = "pass" if (manifest["n_features"] > 0 and not non_numeric_cols and nan_count == 0) else "fail"

    # Write artifacts
    sample_path = f"{outdir}/sample_features.csv"
    desc_path   = f"{outdir}/describe.csv"
    manifest_path = f"{outdir}/feature_manifest.json"
    summary_path  = f"{outdir}/summary.json"

    X.head(20).to_csv(sample_path, index=False)
    X.describe().T.to_csv(desc_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "module": "features",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "data": args.data,
            "features_yaml": args.features,
            "feature_mode": args.feature_mode,
            "feature_groups": groups or [],
            "limit": args.limit,
        },
        "selected": {
            "feature_set_id": spec["id"],
            "n_features": manifest["n_features"],
            "feature_hash": manifest["feature_hash"],
            "first_10": list(X.columns[:10]),
            "rows": int(len(X)),
        },
        "data_quality": {
            "nan_total_after_impute": nan_count,
            "non_numeric_cols_after_clean": non_numeric_cols,
        },
        "pass_criteria": {
            "n_features_gt_0": manifest["n_features"] > 0,
            "no_nans_after_impute": nan_count == 0,
            "all_numeric_after_clean": len(non_numeric_cols) == 0,
        },
        "artifacts": {
            "sample_features_csv": sample_path,
            "describe_csv": desc_path,
            "feature_manifest_json": manifest_path,
            "summary_json": summary_path,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Append to global audit log
    with open("artifacts/diagnostics/audit_log.jsonl", "a") as audit_f:
        audit_f.write(json.dumps(summary) + "\n")

    print(f"[Features] {status.upper()}: {manifest['n_features']} features, hash={manifest['feature_hash']}")
    print(f"Artifacts → {outdir}/")

if __name__ == "__main__":
    main()

#.................................
# Phase-0 single-feature test
python -m src.aaa.diagnostics.features_check \
  --data data_stream/processed/date_2024-01-*/part-*.parquet \
  --features configs/features/fs_test.yaml \
  --feature-mode all \
  --limit 5000


#===============================================
mkdir -p src/aaa/exp
mkdir -p src/aaa/diagnostics
mkdir -p artifacts/diagnostics/model

touch src/__init__.py
touch src/aaa/__init__.py
touch src/aaa/exp/__init__.py
touch src/aaa/diagnostics/__init__.py

# new module + diagnostic
touch src/aaa/exp/model_zoo.py
touch src/aaa/diagnostics/model_zoo_check.py


#................................

# configs/experiments/iforest_cpu.yaml
algo: isolation_forest

params:
  n_estimators: 200          # number of trees in the forest
  max_samples: auto          # number of samples to draw for each base estimator
  contamination: 0.05        # expected fraction of anomalies in data
  max_features: 1.0          # fraction of features to draw for each tree
  bootstrap: false           # whether bootstrap samples are used
  n_jobs: -1                 # use all CPU cores
  random_state: 42           # reproducibility
#................................

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


def build_pipeline(algo: str, params: Dict) -> Tuple[Pipeline, bool]:
    """
    Returns (pipeline, needs_fit_predict)
      - needs_fit_predict=True for LOF (since scores are computed during fit)
    Conventions:
      - We standardize features for LOF & OCSVM.
      - Isolation Forest uses raw features.
    """
    a = algo.lower()

    if a == "isolation_forest":
        model = IsolationForest(**params)
        pipe = Pipeline([("model", model)])
        return pipe, False

    if a == "ocsvm":
        model = OneClassSVM(**params)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        return pipe, False

    if a == "lof":
        # Important: LOF uses fit_predict to compute negative_outlier_factor_
        model = LocalOutlierFactor(**params)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        return pipe, True

    raise ValueError(f"Unknown algo: {algo}")


def anomaly_scores(algo: str, pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Normalize direction so that HIGHER = more anomalous across all algos.
      - IsolationForest: score_samples (higher = more normal) → negate.
      - OCSVM: decision_function (positive = inliers) → negate.
      - LOF: negative_outlier_factor_ (more negative = more outlier) → negate.
    """
    a = algo.lower()

    if a == "isolation_forest":
        raw = pipe["model"].score_samples(X)
        return -raw

    if a == "ocsvm":
        raw = pipe["model"].decision_function(X)
        return -raw

    if a == "lof":
        # requires fit_predict to have run; attribute on the model
        raw = pipe["model"].negative_outlier_factor_
        return -raw

    raise ValueError(a)
    
#...........................

from __future__ import annotations
import argparse, os, json, yaml, numpy as np
from datetime import datetime, timezone

from src.aaa.exp.dataio import load_data
from src.aaa.exp.features import load_feature_spec, resolve_feature_columns, select_features
from src.aaa.exp.model_zoo import build_pipeline, anomaly_scores


def main():
    ap = argparse.ArgumentParser(description="Diagnostic check for model_zoo module")
    ap.add_argument("--data", required=True, help="file/folder/glob")
    ap.add_argument("--features", required=True, help="features YAML (grouped schema)")
    ap.add_argument("--config", required=True, help="experiment YAML (algo + params)")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--bins", type=int, default=30, help="histogram bins")
    args = ap.parse_args()

    outdir = "artifacts/diagnostics/model"
    os.makedirs(outdir, exist_ok=True)

    # load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    algo = cfg["algo"].lower()
    params = cfg.get("params", {})

    # load features/data
    spec = load_feature_spec(args.features)
    cols = resolve_feature_columns(spec, mode="all")
    raw = load_data(args.data, limit=args.limit)
    X = select_features(raw, cols).to_numpy()

    # build + fit
    pipe, needs_fit_predict = build_pipeline(algo, params)
    if needs_fit_predict:
        _ = pipe.fit_predict(X)
    else:
        pipe.fit(X)

    # score & summarize
    scores = anomaly_scores(algo, pipe, X)
    smin, smax, sme, svar = float(np.min(scores)), float(np.max(scores)), float(np.mean(scores)), float(np.var(scores))

    # histogram
    hist_counts, hist_edges = np.histogram(scores, bins=args.bins)
    hist_path = f"{outdir}/score_histogram.csv"
    with open(hist_path, "w") as f:
        f.write("bin_left,bin_right,count\n")
        for i in range(len(hist_counts)):
            f.write(f"{hist_edges[i]},{hist_edges[i+1]},{int(hist_counts[i])}\n")

    # pass criteria
    status = "pass" if (len(scores) > 0 and svar > 0) else "fail"

    summary = {
        "module": "model_zoo",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "algo": algo,
            "params": params,
            "data": args.data,
            "features_yaml": args.features,
            "limit": args.limit,
        },
        "fit": {
            "rows": int(X.shape[0]),
            "features": int(X.shape[1]),
        },
        "scores": {
            "min": smin,
            "max": smax,
            "mean": sme,
            "variance": svar,
        },
        "pass_criteria": {
            "rows_gt_0": X.shape[0] > 0,
            "features_gt_0": X.shape[1] > 0,
            "variance_gt_0": svar > 0.0
        },
        "artifacts": {
            "score_histogram_csv": hist_path
        }
    }

    summary_path = f"{outdir}/summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Append to global audit log
    with open("artifacts/diagnostics/audit_log.jsonl", "a") as audit_f:
        audit_f.write(json.dumps(summary) + "\n")

    print(f"[Model] {status.upper()} — {algo} on {summary['fit']['rows']}x{summary['fit']['features']}, "
          f"score var={svar:.6f}. Artifacts → {outdir}/")

if __name__ == "__main__":
    main()
    
#.............................
python -m src.aaa.diagnostics.model_zoo_check \
  --data data_stream/processed/date_2024-01-*/part-*.parquet \
  --features configs/features/fs_test.yaml \
  --config configs/experiments/iforest_cpu.yaml \
  --limit 3000

#=================================================
mkdir -p src/aaa/exp
mkdir -p src/aaa/diagnostics
mkdir -p artifacts/diagnostics/metrics

touch src/__init__.py
touch src/aaa/__init__.py
touch src/aaa/exp/__init__.py
touch src/aaa/diagnostics/__init__.py

# new module + diagnostic
touch src/aaa/exp/metrics.py
touch src/aaa/diagnostics/metrics_check.py

#............................
from __future__ import annotations
import numpy as np
import pandas as pd


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k highest anomaly scores."""
    if k <= 0:
        return np.array([], dtype=int)
    k = min(k, len(scores))
    return np.argpartition(-scores, k - 1)[:k]


def overlap_at_k(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> float:
    """
    Compare overlap between top-k sets of two score arrays.
    Returns ratio in [0,1].
    """
    idx_a = set(topk_indices(scores_a, k))
    idx_b = set(topk_indices(scores_b, k))
    if not idx_a or not idx_b:
        return 0.0
    return len(idx_a & idx_b) / float(k)


def score_summary(scores: np.ndarray) -> dict:
    """Basic descriptive stats for anomaly scores."""
    return {
        "n": int(len(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "var": float(np.var(scores)),
    }


def attach_scores(df: pd.DataFrame, scores: np.ndarray, id_cols: list[str]) -> pd.DataFrame:
    """Attach anomaly scores to a DataFrame with optional ID columns."""
    out = df[id_cols].copy() if id_cols else pd.DataFrame(index=np.arange(len(scores)))
    out["score"] = scores
    return out
    
    
#.............................::.

from __future__ import annotations
import os, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src.aaa.exp.metrics import topk_indices, overlap_at_k, score_summary, attach_scores


def main():
    outdir = "artifacts/diagnostics/metrics"
    os.makedirs(outdir, exist_ok=True)

    # synthetic scores
    np.random.seed(42)
    scores_a = np.random.randn(100)
    scores_b = np.random.randn(100)

    # test functions
    idx = topk_indices(scores_a, 10)
    overlap = overlap_at_k(scores_a, scores_b, 10)
    summ = score_summary(scores_a)

    df = pd.DataFrame({"device_id": np.arange(100)})
    attached = attach_scores(df, scores_a, ["device_id"])
    attached.head(20).to_csv(f"{outdir}/attached_sample.csv", index=False)

    summary = {
        "module": "metrics",
        "status": "pass" if len(idx) == 10 and 0.0 <= overlap <= 1.0 else "fail",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "topk_len": len(idx),
            "overlap@10": overlap,
            "summary_stats": summ,
        },
        "artifacts": {
            "attached_sample_csv": f"{outdir}/attached_sample.csv"
        }
    }

    summary_path = f"{outdir}/summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Append to audit log
    with open("artifacts/diagnostics/audit_log.jsonl", "a") as audit_f:
        audit_f.write(json.dumps(summary) + "\n")

    print(f"[Metrics] {summary['status'].upper()} — "
          f"topk_len={summary['checks']['topk_len']}, overlap@10={overlap:.2f}")


if __name__ == "__main__":
    main()
    
#................................

python -m src.aaa.diagnostics.metrics_check

#===================================
mkdir -p src/aaa/exp
mkdir -p src/aaa/diagnostics
mkdir -p artifacts/diagnostics/runexp
mkdir -p artifacts/models artifacts/reports artifacts/logs
mkdir -p configs/experiments   # if you haven’t already

touch src/__init__.py
touch src/aaa/__init__.py
touch src/aaa/exp/__init__.py
touch src/aaa/diagnostics/__init__.py

# new module + diagnostic
touch src/aaa/exp/run_experiment.py
touch src/aaa/diagnostics/runexp_check.py

#....................
mkdir -p src/aaa/exp
mkdir -p src/aaa/diagnostics
mkdir -p artifacts/diagnostics/runexp
mkdir -p artifacts/models artifacts/reports artifacts/logs
mkdir -p configs/experiments   # if you haven’t already

touch src/__init__.py
touch src/aaa/__init__.py
touch src/aaa/exp/__init__.py
touch src/aaa/diagnostics/__init__.py

# new module + diagnostic
touch src/aaa/exp/run_experiment.py
touch src/aaa/diagnostics/runexp_check.py

#.............................

from __future__ import annotations
import os, json, time, argparse, yaml
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import mlflow
import joblib

from .dataio import load_data
from .features import load_feature_spec, resolve_feature_columns, select_features, feature_manifest
from .model_zoo import build_pipeline, anomaly_scores
from .metrics import topk_indices, score_summary


def _zscore_baseline(Xdf: pd.DataFrame) -> np.ndarray:
    """Composite z baseline: per-row max absolute z over columns."""
    mu = Xdf.mean(axis=0)
    sd = Xdf.std(axis=0).replace(0, 1e-8)
    z = (Xdf - mu) / sd
    return z.abs().max(axis=1).to_numpy()


def main():
    ap = argparse.ArgumentParser(description="AAA unsupervised experiment runner (MLflow)")
    ap.add_argument("--data", required=True, help="file/folder/glob path to data")
    ap.add_argument("--features", required=True, help="features YAML (grouped schema)")
    ap.add_argument("--config", required=True, help="experiment YAML (algo + params)")
    ap.add_argument("--id-cols", default=None, help="comma list of ID cols to carry (overrides YAML i_columns)")
    ap.add_argument("--feature-mode", choices=["all","groups","list"], default="all")
    ap.add_argument("--feature-groups", default=None, help="comma list when mode=groups (e.g., SC,SS)")
    ap.add_argument("--feature-list", default=None, help="comma list when mode=list")
    ap.add_argument("--limit", type=int, default=None, help="row cap for quick runs")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    args = ap.parse_args()

    # Ensure artifact dirs
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/reports", exist_ok=True)
    os.makedirs("artifacts/logs", exist_ok=True)

    # Load experiment config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    algo = cfg["algo"].lower()
    params = cfg.get("params", {})

    # Load feature spec + resolve columns
    spec = load_feature_spec(args.features)
    if args.id_cols:
        id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    else:
        id_cols = list(spec.get("i_columns", []))

    groups = [g.strip() for g in args.feature_groups.split(",")] if args.feature_groups else None
    explicit = [c.strip() for c in args.feature_list.split(",")] if args.feature_list else None
    cols = resolve_feature_columns(spec, mode=args.feature_mode, groups=groups, extra=explicit)

    # Load data
    t0 = time.time()
    raw = load_data(args.data, limit=args.limit)
    load_s = time.time() - t0

    # ID frame
    id_frame = pd.DataFrame()
    for c in id_cols:
        if c in raw.columns:
            id_frame[c] = raw[c]
    if id_frame.empty:
        id_frame["row_id"] = np.arange(len(raw))

    # Select/clean features
    Xdf = select_features(raw, cols)
    manifest = feature_manifest(list(Xdf.columns))

    # Build pipeline and fit
    pipe, needs_fit_predict = build_pipeline(algo, params)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(f"AAA-Unsupervised/{spec['id']}/{algo}")

    with mlflow.start_run(run_name=f"{algo}-{spec['id']}"):
        # Params
        mlflow.log_params({
            "algo": algo,
            "feature_set": spec["id"],
            "feature_mode": args.feature_mode,
            "feature_groups": ",".join(groups) if groups else "",
            "n_features": manifest["n_features"],
            **params,
        })
        mlflow.log_dict(manifest, f"reports/feature_manifest_{spec['id']}.json")

        # Fit
        t1 = time.time()
        X = Xdf.to_numpy()
        if needs_fit_predict:
            _ = pipe.fit_predict(X)
        else:
            pipe.fit(X)
        fit_s = time.time() - t1

        # Score
        scores = anomaly_scores(algo, pipe, X)
        summ = score_summary(scores)
        mlflow.log_metrics({
            "fit_time_s": fit_s,
            "rows": int(X.shape[0]),
            "score_min": summ["min"],
            "score_max": summ["max"],
            "score_mean": summ["mean"],
            "score_var": summ["var"],
            "load_time_s": load_s,
        })

        # Baseline overlap vs zscore@K
        zscores = _zscore_baseline(Xdf)
        K = int(args.topk)
        a_idx = topk_indices(scores, K)
        z_idx = topk_indices(zscores, K)
        overlap = len(set(a_idx.tolist()) & set(z_idx.tolist())) / float(max(1, K))
        mlflow.log_metric("overlap_vs_zscore@K", overlap)

        # Export Top-K
        top_df = id_frame.iloc[a_idx].copy()
        top_df["score"] = scores[a_idx]
        top_path = f"artifacts/reports/topk_{algo}_{spec['id']}.csv"
        top_df.to_csv(top_path, index=False)
        mlflow.log_artifact(top_path)

        # Score histogram artifact
        hist_counts, hist_edges = np.histogram(scores, bins=30)
        hist_path = f"artifacts/reports/score_hist_{algo}_{spec['id']}.csv"
        with open(hist_path, "w") as f:
            f.write("bin_left,bin_right,count\n")
            for i in range(len(hist_counts)):
                f.write(f"{hist_edges[i]},{hist_edges[i+1]},{int(hist_counts[i])}\n")
        mlflow.log_artifact(hist_path)

        # Save model
        model_path = f"artifacts/models/{algo}_{spec['id']}.joblib"
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(model_path)

        # Run meta log
        meta = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "algo": algo,
            "feature_set": spec["id"],
            "n_features": manifest["n_features"],
            "rows": int(X.shape[0]),
            "fit_time_s": fit_s,
            "topk": K,
            "overlap_vs_zscore@K": overlap,
            "data_path": args.data,
            "features_yaml": args.features,
            "config_yaml": args.config,
        }
        with open("artifacts/logs/run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact("artifacts/logs/run_meta.json")

    print(
        f"Run complete: algo={algo}, fs={spec['id']}, n={len(Xdf)} rows, "
        f"fit_time={fit_s:.2f}s. Top-K → {top_path}"
    )


if __name__ == "__main__":
    main()
    
#...................................
from __future__ import annotations
import os, json, subprocess, shlex
from datetime import datetime, timezone

def main():
    outdir = "artifacts/diagnostics/runexp"
    os.makedirs(outdir, exist_ok=True)

    # Edit these two paths for your environment if needed:
    data = "data_stream/processed/date_2024-01-*/part-*.parquet"
    features_yaml = "configs/features/fs_test.yaml"
    config_yaml = "configs/experiments/iforest_cpu.yaml"

    cmd = (
        "python -m src.aaa.exp.run_experiment "
        f"--data {shlex.quote(data)} "
        f"--features {shlex.quote(features_yaml)} "
        f"--config {shlex.quote(config_yaml)} "
        "--limit 3000 --topk 50"
    )
    print(f"[RunExpCheck] Launching: {cmd}")
    ret = subprocess.run(cmd, shell=True)
    status = "pass" if ret.returncode == 0 else "fail"

    # Verify expected artifacts exist
    expected = [
        "artifacts/reports",
        "artifacts/models",
        "mlruns",  # MLflow backend dir
    ]
    exists = {p: os.path.exists(p) for p in expected}

    summary = {
        "module": "run_experiment",
        "status": "pass" if (status == "pass" and all(exists.values())) else "fail",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "data": data,
            "features_yaml": features_yaml,
            "config_yaml": config_yaml,
            "limit": 3000,
            "topk": 50,
        },
        "checks": exists,
        "artifacts_hint": {
            "reports_dir": "artifacts/reports",
            "models_dir": "artifacts/models",
            "mlflow_dir": "mlruns"
        }
    }
    with open(f"{outdir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Append to global audit log
    with open("artifacts/diagnostics/audit_log.jsonl", "a") as audit_f:
        audit_f.write(json.dumps(summary) + "\n")

    print(f"[RunExpCheck] {summary['status'].upper()} — see {outdir}/summary.json")

if __name__ == "__main__":
    main()
    
#...............................
# optional: start MLflow UI in another terminal
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# run the smoke check
python -m src.aaa.diagnostics.runexp_check

#................................
python -m src.aaa.exp.run_experiment \
  --data data_stream/processed/date_2024-01-*/part-*.parquet \
  --features configs/features/fs_test.yaml \
  --config configs/experiments/iforest_cpu.yaml \
  --limit 3000 \
  --topk 50

#=========================================
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:////Users/bells1/Projects/AAA_Pipeline_Practice/mlflow.db")
client = MlflowClient()

# list deleted, find your experiment
for e in client.search_experiments(view_type=mlflow.entities.ViewType.DELETED_ONLY):
    print(e.experiment_id, e.name)

# restore by ID
client.restore_experiment("<ID>")
