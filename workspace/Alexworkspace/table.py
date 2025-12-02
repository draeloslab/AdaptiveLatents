import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def _load_runs(jsonl_path):
    """Return list of runs; each run is a dict of lists keyed by metric name."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    runs = []
    cur = {}
    for r in rows:
        if int(r.get("eval", -1)) == 0 and cur:
            runs.append(cur); cur = {}
        for k, v in r.items():
            cur.setdefault(k, []).append(v)
    if cur:
        runs.append(cur)
    return runs


def _final_angles(runs, use_best=False):
    """Final (or best) angle per run."""
    out = []
    for r in runs:
        ang = np.array(r.get("angle_deg", []), dtype=float)
        if ang.size == 0:
            out.append(np.nan)
        else:
            out.append(np.nanmin(ang) if use_best else ang[-1])
    return np.array(out, dtype=float)


def make_results_table(
    conditions,                 # dict/OrderedDict or list of (label, path)
    paired_baseline_label,      # which label is the baseline
    use_best=False,             # False=last angle, True=min angle in run
    success_thresh=90.0,        # “success” if angle<threshold
    save_csv=None               # path to save CSV (optional)
):
    """
    Returns (table_df, summary_df).
    Table: one row per run index, columns per condition + deltas vs baseline.
    Summary: per-condition n, median, IQR, success rate, and Wilcoxon p (vs baseline).
    """
    if isinstance(conditions, dict):
        conditions = list(conditions.items())
    else:
        conditions = list(conditions)

    # Load angles per condition
    angles_by_label = {}
    for label, path in conditions:
        runs = _load_runs(Path(path))
        angles_by_label[label] = _final_angles(runs, use_best=use_best)

    # Align by min run count
    min_len = min(len(a) for a in angles_by_label.values())
    if min_len == 0:
        raise ValueError("No complete runs found across conditions.")
    for k in angles_by_label:
        angles_by_label[k] = angles_by_label[k][:min_len]

    # Build table: one row per run index
    df = pd.DataFrame({lbl: angles_by_label[lbl] for lbl, _ in conditions})
    df.index.name = "run_idx"

    # Deltas vs baseline
    base = df[paired_baseline_label].to_numpy()
    eps = 1e-9  # for safe % change
    for lbl, _ in conditions:
        if lbl == paired_baseline_label:
            continue
        delta = df[lbl] - df[paired_baseline_label]
        pct = np.where(np.abs(base) > eps, 100.0 * delta / base, np.nan)
        df[f"Δ({lbl}−{paired_baseline_label}) [deg]"] = delta
        df[f"%Δ({lbl}−{paired_baseline_label})"] = pct

    # Summary stats per condition
    rows = []
    for lbl, _ in conditions:
        vals = df[lbl].to_numpy()
        vals_f = vals[np.isfinite(vals)]
        if vals_f.size == 0:
            rows.append(dict(
                condition=lbl, n=0, median=np.nan, iqr_low=np.nan, iqr_high=np.nan,
                success_rate=np.nan, wilcoxon_p=np.nan, median_delta_vs_baseline=np.nan, median_pct_vs_baseline=np.nan
            ))
            continue

        med = np.median(vals_f)
        q1, q3 = np.percentile(vals_f, [25, 75])
        succ = float(np.mean(vals_f < success_thresh))

        if lbl == paired_baseline_label:
            rows.append(dict(
                condition=lbl, n=len(vals_f), median=med, iqr_low=q1, iqr_high=q3,
                success_rate=succ, wilcoxon_p=np.nan,
                median_delta_vs_baseline=np.nan, median_pct_vs_baseline=np.nan
            ))
        else:
            delta = df[lbl].to_numpy() - df[paired_baseline_label].to_numpy()
            delta_f = delta[np.isfinite(delta)]
            med_delta = np.median(delta_f)

            pct = np.where(np.abs(base) > eps, 100.0 * delta / base, np.nan)
            med_pct = np.nanmedian(pct)

            pval = np.nan
            if HAVE_SCIPY:
                try:
                    # Paired, two-sided
                    stat, pval = wilcoxon(delta_f, alternative="two-sided", zero_method="wilcox")
                except ValueError:
                    pval = np.nan

            rows.append(dict(
                condition=lbl, n=len(vals_f), median=med, iqr_low=q1, iqr_high=q3,
                success_rate=succ, wilcoxon_p=pval,
                median_delta_vs_baseline=med_delta, median_pct_vs_baseline=med_pct
            ))

    summary = pd.DataFrame(rows).set_index("condition")

    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=True)
        print(f"Saved table to: {save_csv}")

    # Console summary
    print("\n=== Per-condition summary (angles in degrees) ===")
    print(summary.to_string(float_format=lambda x: f"{x:.3g}"))

    return df, summary


# ---------------- Example ----------------
conditions = {
    "no-perturb": r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\nopert.jsonl",
    "pert=0.5": r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\pert0_5.jsonl",
    "pert=0.7": r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\pert0_7.jsonl",
    "pert=0.8": r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\pert0_8.jsonl",
    "pert=0.9": r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\pert0_9.jsonl",
}
table, summary = make_results_table(
    conditions,
    paired_baseline_label="no-perturb",
    use_best=False,             # stick with final angle
    success_thresh=90.0,
    save_csv=r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\angletab.csv"
)
print(table.head())
