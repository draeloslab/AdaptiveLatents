import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch 

CURRENT_SESSION_INFO_PATH = r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\metricsfolder\session_id.txt"


def load_current_session_info(info_path=CURRENT_SESSION_INFO_PATH):
    """
    Load current session info from a text file.

    Assumption:
      - The txt file contains ONLY the session_id as a single line, e.g.
            session_20251124_200445

    The JSONL metrics file is then inferred as:
        metrics_{session_id}.jsonl

    in the SAME DIRECTORY as the txt file.
    """
    info_path = Path(info_path)
    if not info_path.exists():
        raise FileNotFoundError(f"Session info file not found: {info_path}")

    text = info_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Session info file is empty: {info_path}")

    # Entire content is just the session_id
    session_id = text.splitlines()[0].strip()

    # JSONL is inferred as metrics_{session_id}.jsonl next to the txt file
    jsonl_path = info_path.parent / f"metrics_{session_id}.jsonl"

    return {"session_id": session_id, "jsonl_path": str(jsonl_path)}


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------

def load_run_summaries(jsonl_path, session_id=None):
    """
    Load only the *summary* rows from your JSONL file.

    Summary rows are the ones you wrote at the end of each run, which contain:
      - run_id
      - optimizer
      - total_time
      - final_angle_deg
      - nnz
      - success_lt_90
      - success_lt_45

    If session_id is given, we filter to rows with that session_id.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No such file: {jsonl_path}")

    summaries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if session_id is not None:
                if row.get("session_id") != session_id:
                    continue

            # Heuristic: summary rows have 'total_time', per-eval rows have 'eval'
            if "total_time" in row:
                summaries.append(row)

    if not summaries:
        raise ValueError("No summary rows with 'total_time' found in JSONL (after session filter, if any).")
    return summaries


# ---------------------------------------------------------------------
# Plot runs with session-wide style organization
# ---------------------------------------------------------------------

def plot_recent_runs(
    jsonl_path,
    alignment_metric="angle",
    x_mode="relative",
    session_id=None,
    save_path=None,
    dpi=120,
    num_runs=None,  # if None => plot ALL runs in the session
):
    """
    Plot runs for the given JSONL + session_id.

    Panels: cosine similarity, alignment (angle or projection), objective, reg term.

    Styling rules:
      - Entries with the same script_run_id share the same marker.
      - For each script_run_id, its i-th run gets color[i].
        The same i-th position in another script_run_id gets the same color index.

    If num_runs is None (default), we use ALL runs in the session.
    Otherwise, we use the last `num_runs` runs.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No such file: {jsonl_path}")

    # Load all rows, then filter by session_id if provided
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if session_id is not None:
                if r.get("session_id") != session_id:
                    continue
            rows.append(r)

    summary_rows = [r for r in rows if "total_time" in r]
    eval_rows = [r for r in rows if "eval" in r]
    summary_by_run = {r["run_id"]: r for r in summary_rows}
    if not summary_rows:
        raise ValueError("No summary rows found — run logging incomplete or wrong session_id.")

    # -----------------------------------------------------------------
    # Build style map: marker by script_run_id, color by "run index" per script
    # -----------------------------------------------------------------
    runs_by_script = {}  # script_run_id -> [run_id1, run_id2, ...] in chronological order
    for s in summary_rows:
        sid = s.get("script_run_id", "unknown_script")
        rid = s["run_id"]
        runs_by_script.setdefault(sid, []).append(rid)

    unique_script_ids = list(runs_by_script.keys())

    MARKERS = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
    marker_for_script = {
        sid: MARKERS[i % len(MARKERS)] for i, sid in enumerate(unique_script_ids)
    }

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = [f"C{i}" for i in range(10)]

    # style_map: run_id -> {marker, color, script_run_id}
    style_map = {}
    for sid, rids in runs_by_script.items():
        for idx, rid in enumerate(rids):
            color = color_cycle[idx % len(color_cycle)]
            style_map[rid] = {
                "script_run_id": sid,
                "marker": marker_for_script[sid],
                "color": color,
            }

    # -----------------------------------------------------------------
    # Choose which run_ids to plot
    # -----------------------------------------------------------------
    ordered_run_ids = [r["run_id"] for r in summary_rows]  # chronological
    if num_runs is None:
        selected_run_ids = ordered_run_ids  # ALL runs in this session
    else:
        selected_run_ids = ordered_run_ids[-num_runs:]

    # Group eval rows by run_id (only those we will plot)
    runs = {}
    for r in eval_rows:
        rid = r.get("run_id", None)
        if rid in selected_run_ids:
            runs.setdefault(rid, []).append(r)

    if not runs:
        raise ValueError("No matching eval rows for selected run IDs.")

    # Sort within each run by eval index
    for rid in runs:
        runs[rid] = sorted(runs[rid], key=lambda r: r["eval"])

    # Choose alignment key
    if alignment_metric == "angle":
        align_key = "angle_deg"
        align_label = "Angle (deg, lower better)"
    elif alignment_metric == "proj":
        align_key = "align_proj"
        align_label = "Alignment (projection, higher better)"
    else:
        raise ValueError("alignment_metric must be 'angle' or 'proj'")

    # Figure layout
    fig, axes = plt.subplots(2, 2, figsize=(10, 10),
                             sharex=(x_mode != "eval"), constrained_layout=True)
    ax2 = axes[0, 0]  # alignment
    ax3 = axes[1, 0]  # obj
    ax1 = axes[0, 1]  # cosine
    ax4 = axes[1, 1]  # reg term

    # Plot each detected run
    for i, rid in enumerate(selected_run_ids, 1):
        if rid not in runs:
            continue
        block = runs[rid]
        evals = np.array([b.get("eval", np.nan) for b in block])
        cos = np.array([b.get("align_cos", np.nan) for b in block])
        align = np.array([b.get(align_key, np.nan) for b in block])
        obj = np.array([b.get("obj", np.nan) for b in block])
        reg = np.array([b.get("reg term", np.nan) for b in block])

        x = np.arange(len(block)) if x_mode == "relative" else evals

        summary = summary_by_run.get(rid, {})
        optimizer = summary.get("optimizer", "unknown")

        label = f"run {i} ({optimizer})"

        style = style_map.get(rid, {})
        marker = style.get("marker", "o")
        color = style.get("color", None)

        ax1.plot(x, cos, marker=marker, linewidth=1, label=label,
                 markersize=2, color=color)
        ax2.plot(x, align, marker=marker, linewidth=1, label=label,
                 markersize=2, color=color)
        ax3.plot(x, obj, marker=marker, linewidth=1, label=label,
                 markersize=2, color=color)
        ax4.plot(x, reg, marker=marker, linewidth=1, label=label,
                 markersize=2, color=color)

        def annotate_last(axis, xdata, ydata):
            if len(xdata) > 0 and len(ydata) > 0:
                xf = xdata[-1]
                yf = ydata[-1]
                axis.text(
                    xf, yf,
                    f"{yf:.5g}",
                    fontsize=7,
                    color="black",
                    ha="left",
                    va="center"
                )

        # annotate_last(ax2, x, align)

    # Axis formatting
    ax1.set_ylabel("Cosine (higher better)")
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', useOffset=False)
    ax1.yaxis.get_major_formatter().set_scientific(False)

    ax2.set_ylabel(align_label)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='plain', useOffset=False)
    ax2.yaxis.get_major_formatter().set_scientific(False)

    ax3.set_ylabel("Objective (lower better)")
    ax3.set_xlabel("Evaluation step" if x_mode == "eval" else "Relative step")
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='plain', useOffset=False)
    ax3.yaxis.get_major_formatter().set_scientific(False)

    ax4.set_ylabel("Reg term")
    ax4.set_xlabel("Evaluation step" if x_mode == "eval" else "Relative step")
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(style='plain', useOffset=False)
    ax4.yaxis.get_major_formatter().set_scientific(False)

    fig.suptitle(
        f"Session {session_id} – {len(selected_run_ids)} runs ({jsonl_path.name})",
        fontsize=14
    )
    handles, labels = ax1.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    fig.legend(handles, labels, title="Run order", loc="upper right")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        print(f"Saved plot to: {save_path}")

    plt.show()
    return selected_run_ids, style_map


# ---------------------------------------------------------------------
# Stats + success rates
# ---------------------------------------------------------------------

def compute_final_angle_stats(jsonl_path, selected_run_ids, session_id=None):
    summaries = load_run_summaries(jsonl_path, session_id=session_id)

    selected = [r for r in summaries if r["run_id"] in selected_run_ids]
    angles = np.array([r["final_angle_deg"] for r in selected], dtype=float)

    return {
        "n_runs": len(angles),
        "mean": float(np.mean(angles)),
        "std": float(np.std(angles, ddof=1) if len(angles) > 1 else 0.0),
        "angles": angles.tolist()
    }


def save_mean_std(stats_dict, out_path):
    """
    Save only mean + std (plus timestamp for tracking) to a separate JSONL file.
    """
    row = {
        "n_runs": stats_dict["n_runs"],
        "mean_angle_deg": stats_dict["mean"],
        "std_angle_deg": stats_dict["std"],
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    print(f"📁 Saved mean/std summary → {out_path}")
    print(row)

def compute_success_rates_by_script_run(
    jsonl_path, selected_run_ids, thresholds=(90.0, 45.0), session_id=None
):
    """
    Compute success rates separately for each *script_run_id*.

    Returns:
      {
        script_run_id: {
           "n_runs": int,
           "angles": [...],
           "rates": {thr: rate, ...},
           "optimizer": "<name>",
        },
        ...
      }
    """
    summaries = load_run_summaries(jsonl_path, session_id=session_id)

    # filter to the chosen runs
    selected = [r for r in summaries if r["run_id"] in selected_run_ids]
    if not selected:
        raise ValueError("No summary rows match the selected run_ids.")

    # group by script_run_id
    by_script = {}
    for r in selected:
        sid = r.get("script_run_id", "unknown_script")
        by_script.setdefault(sid, []).append(r)

    result = {}
    for sid, rows in by_script.items():
        angles = np.array([row["final_angle_deg"] for row in rows], dtype=float)
        rates = {}
        for thr in thresholds:
            rates[thr] = float(np.mean(angles < thr))

        # assume all rows in this script_run_id use the same optimizer
        opt_names = {row.get("optimizer", "unknown") for row in rows}
        if len(opt_names) == 1:
            opt = next(iter(opt_names))
        else:
            opt = ", ".join(sorted(opt_names))

        result[sid] = {
            "n_runs": len(angles),
            "angles": angles.tolist(),
            "rates": rates,
            "optimizer": opt,
        }

    print("=== Success rates by script_run_id ===")
    for sid, stats in result.items():
        print(f"\nscript_run_id: {sid}")
        print(f"  optimizer: {stats['optimizer']}")
        print(f"  n_runs: {stats['n_runs']}")
        for thr, rate in stats["rates"].items():
            print(f"  frac final_angle < {thr}°: {rate:.3f}")

    return result


def plot_success_rates_by_script_run(success_stats, title="Success rates (final angle)"):
    """
    success_stats is the dict returned by compute_success_rates_by_script_run.
    Creates one bar chart per script_run_id in a single figure.
    """
    script_ids = sorted(success_stats.keys())
    n_scripts = len(script_ids)
    thresholds = sorted(next(iter(success_stats.values()))["rates"].keys())

    fig, axes = plt.subplots(
        1, n_scripts, figsize=(4 * n_scripts, 4), sharey=True, squeeze=False
    )
    axes = axes[0]  # 1 x n_scripts

    for ax, sid in zip(axes, script_ids):
        stats = success_stats[sid]
        rates = stats["rates"]
        optimizer = stats["optimizer"]

        values = [rates[t] for t in thresholds]
        x = np.arange(len(thresholds))

        ax.bar(x, values, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"< {int(t)}°" for t in thresholds])
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{optimizer}\n{sid[:8]}")  # show optimizer + short script id
        ax.set_ylabel("Success fraction")

        for xi, yi in zip(x, values):
            ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()



def get_optimizer_map(jsonl_path, selected_run_ids, session_id=None):
    """
    Return a dict run_id -> optimizer for the selected runs.
    """
    summaries = load_run_summaries(jsonl_path, session_id=session_id)
    by_id = {r["run_id"]: r for r in summaries}
    optimizer_map = {}
    for rid in selected_run_ids:
        if rid in by_id:
            optimizer_map[rid] = by_id[rid].get("optimizer", "unknown")
    return optimizer_map

def get_script_run_map(jsonl_path, selected_run_ids, session_id=None):
    """
    Return a dict run_id -> script_run_id for the selected runs.
    """
    summaries = load_run_summaries(jsonl_path, session_id=session_id)
    by_id = {r["run_id"]: r for r in summaries}
    script_run_map = {}
    for rid in selected_run_ids:
        if rid in by_id:
            script_run_map[rid] = by_id[rid].get("script_run_id", "unknown_script")
    return script_run_map

# ---------------------------------------------------------------------
# Time vs angle / sparsity / times
# ---------------------------------------------------------------------

def get_time_vs_angle(jsonl_path, selected_run_ids, session_id=None):
    """
    Extract (total_time, final_angle_deg) for selected runs.
    Returns a dict with arrays for plotting and the ordered run_ids.
    """
    summaries = load_run_summaries(jsonl_path, session_id=session_id)
    selected = [r for r in summaries if r["run_id"] in selected_run_ids]

    if not selected:
        raise ValueError("No matching runs for given run_ids.")

    # Preserve the order of selected_run_ids
    by_id = {r["run_id"]: r for r in selected}
    ordered = [by_id[rid] for rid in selected_run_ids if rid in by_id]

    times = np.array([r["total_time"] for r in ordered], dtype=float)
    angles = np.array([r["final_angle_deg"] for r in ordered], dtype=float)
    run_ids = [r["run_id"] for r in ordered]

    print(f"=== Time vs Angle for {len(times)} runs ===")
    for t, a, rid in zip(times, angles, run_ids):
        print(f"run {rid[:6]}  time={t:.3f}s  →  final_angle={a:.2f}°")

    return {"times": times, "angles": angles, "n_runs": len(times), "run_ids": run_ids}


def plot_time_vs_angle(time_angle_data, style_map=None, script_run_map=None,
                       title="Speed vs Final Alignment"):
    times = time_angle_data["times"]
    angles = time_angle_data["angles"]
    run_ids = time_angle_data.get("run_ids", None)

    fig, ax = plt.subplots(figsize=(6, 5))

    # ------------------------------------------------------------------
    # Scatter points (color per run, marker per script_run_id via style_map)
    # ------------------------------------------------------------------
    if style_map is not None and run_ids is not None:
        for t, a, rid in zip(times, angles, run_ids):
            style = style_map.get(rid, {})
            color = style.get("color", None)
            marker = style.get("marker", "o")
            ax.scatter(t, a, s=55, alpha=0.8, color=color, marker=marker)
    else:
        ax.scatter(times, angles, s=45, alpha=0.8)

    ax.set_xlabel("Optimization Time (seconds)")
    ax.set_ylabel("Final Angle (deg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Legend: one entry per script_run_id (marker matches points)
    # ------------------------------------------------------------------
    if run_ids is not None and script_run_map is not None:
        label_entries = {}   # script_run_id -> (marker, label_text)

        for rid in run_ids:
            script_id = script_run_map.get(rid, "unknown")  
            opt = optimizer_map.get(rid, "unknown") if optimizer_map else "unknown"
            marker = style_map.get(rid, {}).get("marker", "o")

            label = f"{opt} ({script_id[:5]})"
            label_entries[script_id] = (marker, label)

        legend_handles = []
        legend_labels = []

        # Keep deterministic ordering (alphabetical by script id)
        for script_id in sorted(label_entries.keys()):
            marker, label = label_entries[script_id]
            h = ax.scatter([], [], marker=marker, color="black", s=60)
            legend_handles.append(h)
            legend_labels.append(label)

        if legend_handles:
            ax.legend(legend_handles, legend_labels,
                      title="Optimizer / Script Run", loc="best")

    plt.show()



def extract_sparsity(jsonl_path, selected_run_ids, session_id=None):
    """
    Extract nnz (nonzero count) for each selected run.
    """
    summaries = load_run_summaries(jsonl_path, session_id=session_id)
    selected = [r for r in summaries if r["run_id"] in selected_run_ids]

    if not selected:
        raise ValueError("No matching summary entries for nnz metric.")

    # Preserve order of selected_run_ids
    by_id = {r["run_id"]: r for r in selected}
    ordered = [by_id[rid] for rid in selected_run_ids if rid in by_id]

    nnz_values = np.array([r.get("nnz", np.nan) for r in ordered], dtype=float)
    run_ids = [r["run_id"] for r in ordered]

    result = {
        "n_runs": len(nnz_values),
        "nnz": nnz_values.tolist(),
        "mean": float(np.nanmean(nnz_values)),
        "std": float(np.nanstd(nnz_values, ddof=1) if len(nnz_values) > 1 else 0.0),
        "run_ids": run_ids,
    }

    print("\n=== Sparsity Stats ===")
    print(f"Runs analyzed: {result['n_runs']}")
    print(f"Sparsity values (nnz): {result['nnz']}")
    print(f"Mean nnz: {result['mean']:.3f}")
    print(f"Std dev:  {result['std']:.3f}")

    return result


def plot_sparsity_hist(data, title="Sparsity Distribution (nnz count)"):
    nnz_vals = data["nnz"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nnz_vals, bins=min(10, max(3, len(nnz_vals)//2)), alpha=0.7)

    ax.set_xlabel("Number of nonzero neurons (nnz)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    ax.grid(alpha=0.3)
    plt.show()


def plot_sparsity_vs_angle(jsonl_path, selected_run_ids, style_map=None,
                           script_run_map=None, session_id=None,
                           title="Sparsity vs Final Angle"):
    summaries = load_run_summaries(jsonl_path, session_id=session_id)
    selected = [r for r in summaries if r["run_id"] in selected_run_ids]

    if not selected:
        raise ValueError("No matching runs for sparsity vs angle.")

    by_id = {r["run_id"]: r for r in selected}
    ordered = [by_id[rid] for rid in selected_run_ids if rid in by_id]

    nnz = np.array([r["nnz"] for r in ordered], dtype=float)
    angles = np.array([r["final_angle_deg"] for r in ordered], dtype=float)
    run_ids = [r["run_id"] for r in ordered]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Scatter points (color per run, marker per script_run_id)
    if style_map is not None:
        for n, a, rid in zip(nnz, angles, run_ids):
            style = style_map.get(rid, {})
            color = style.get("color", None)
            marker = style.get("marker", "o")
            ax.scatter(n, a, s=55, alpha=0.8, color=color, marker=marker)
    else:
        ax.scatter(nnz, angles, s=55, alpha=0.8)

    ax.set_xlabel("nnz (number of non zeros)")
    ax.set_ylabel("Final Angle (deg)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    # Legend: one entry per script_run_id
    if run_ids is not None and script_run_map is not None:
        label_entries = {}   # script_run_id -> (marker, label_text)

        for rid in run_ids:
            script_id = script_run_map.get(rid, "unknown")  
            opt = optimizer_map.get(rid, "unknown") if optimizer_map else "unknown"
            marker = style_map.get(rid, {}).get("marker", "o")

            label = f"{opt} ({script_id[:5]})"
            label_entries[script_id] = (marker, label)

        legend_handles = []
        legend_labels = []

        # Keep deterministic ordering (alphabetical by script id)
        for script_id in sorted(label_entries.keys()):
            marker, label = label_entries[script_id]
            h = ax.scatter([], [], marker=marker, color="black", s=60)
            legend_handles.append(h)
            legend_labels.append(label)

        if legend_handles:
            ax.legend(legend_handles, legend_labels,
                      title="Optimizer / Script Run", loc="best")

    plt.show()



def get_times(jsonl_path, selected_run_ids, session_id=None):
    """
    Returns a dict: { "times": [ ... ], "run_ids": [ ... ] }
    for the given run_ids.
    """
    summaries = load_run_summaries(jsonl_path, session_id=session_id)
    selected = [r for r in summaries if r["run_id"] in selected_run_ids]

    if not selected:
        raise ValueError("No matching runs for time metric.")

    by_id = {r["run_id"]: r for r in selected}
    ordered = [by_id[rid] for rid in selected_run_ids if rid in by_id]

    times = [r["total_time"] for r in ordered]
    run_ids = [r["run_id"] for r in ordered]

    return {"times": times, "run_ids": run_ids}


# def plot_times_bar(times_data, style_map=None, title="Optimization Time Per Run"):
#     """
#     Bar plot where each bar represents one run's optimization duration.
#     times_data: dict with keys "times" and "run_ids".
#     """
#     times = times_data["times"]
#     run_ids = times_data.get("run_ids", None)

#     fig, ax = plt.subplots(figsize=(6, 4))

#     x = np.arange(len(times))

#     if style_map is not None and run_ids is not None:
#         bar_colors = []
#         for rid in run_ids:
#             style = style_map.get(rid, {})
#             bar_colors.append(style.get("color", None))
#     else:
#         bar_colors = None

#     bars = ax.bar(x, times, alpha=0.75, color=bar_colors)

#     # Label bars
#     for i, t in enumerate(times):
#         ax.text(i, t, f"{t:.2f}s", ha="center", va="bottom", fontsize=8)

#     ax.set_xticks(x)
#     ax.set_xticklabels([f"{i+1}" for i in range(len(times))])

#     ax.set_ylabel("Time (seconds)")
#     ax.set_xlabel("Run Index (session order)")
#     ax.set_title(title)

#     ax.grid(axis="y", alpha=0.3)
#     plt.tight_layout()
#     plt.show()

def plot_times_bar(times_data,
                   style_map=None,
                   optimizer_map=None,
                   script_run_map=None,
                   title="Optimization Time Per Run"):
    """
    Bar plot where each bar represents one run's optimization duration.

    Bars are colored by script_run_id (all runs from the same script_run_id
    get the same color). Legend entries are:

        optimizer (scriptID[:5])
    """
    times = times_data["times"]
    run_ids = times_data.get("run_ids", None)

    fig, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(len(times))

    # ------------------------------------------------------------------
    # Assign a color per script_run_id
    # ------------------------------------------------------------------
    bar_colors = None
    script_ids = None

    if style_map is not None and run_ids is not None:
        # determine script_run_id for each run
        script_ids = []
        for rid in run_ids:
            if script_run_map is not None:
                sid = script_run_map.get(rid, "unknown_script")
            else:
                sid = style_map.get(rid, {}).get("script_run_id", "unknown_script")
            script_ids.append(sid)

        # unique script IDs in order of appearance
        unique_scripts = []
        for sid in script_ids:
            if sid not in unique_scripts:
                unique_scripts.append(sid)

        # color cycle
        base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not base_colors:
            base_colors = [f"C{i}" for i in range(10)]

        # map script_run_id -> color
        color_for_script = {}
        for i, sid in enumerate(unique_scripts):
            color_for_script[sid] = base_colors[i % len(base_colors)]

        # color each bar based on its script_run_id
        bar_colors = [color_for_script[sid] for sid in script_ids]
    else:
        color_for_script = {}
        unique_scripts = []

    bars = ax.bar(x, times, alpha=0.75, color=bar_colors)

    # Label bars with time
    for i, t in enumerate(times):
        ax.text(i, t, f"{t:.4f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}" for i in range(len(times))])

    ax.set_ylabel("Time (seconds)")
    ax.set_xlabel("Run Index (session order)")
    ax.set_title(title)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Legend: color → "optimizer (scriptID[:5])"
    # ------------------------------------------------------------------
    if script_ids is not None and optimizer_map is not None:
        legend_handles = []
        legend_labels = []

        for sid in unique_scripts:
            color = color_for_script[sid]

            # find one run_id belonging to this script to read its optimizer
            opt = "unknown"
            for rid in run_ids:
                sid_r = (script_run_map.get(rid, "unknown_script")
                         if script_run_map is not None
                         else style_map.get(rid, {}).get("script_run_id", "unknown_script"))
                if sid_r == sid:
                    opt = optimizer_map.get(rid, "unknown")
                    break

            label = f"{opt} ({sid[:5]})"
            legend_handles.append(Patch(facecolor=color, edgecolor="black", label=label))
            legend_labels.append(label)

        if legend_handles:
            ax.legend(legend_handles, legend_labels,
                      title="script_run_id / optimizer", loc="best")

    plt.show()




# ---------------------------------------------------------------------
# Main usage: plot the whole session
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Load current session info and JSONL location
    session_info = load_current_session_info()
    current_session_id = session_info["session_id"]
    jsonl_loc = session_info["jsonl_path"]

    

    # 1) Plot ALL runs for this session + get their run_ids and style map
    session_run_ids, style_map = plot_recent_runs(
        jsonl_loc,
        alignment_metric="angle",
        x_mode="eval",
        session_id=current_session_id,
        num_runs=None,  # None means "all runs in this session"
    )

    # 2) Compute mean / std final angle over ALL those runs
    stats = compute_final_angle_stats(jsonl_loc, session_run_ids, session_id=current_session_id)
    print(stats)

    success_stats_by_script = compute_success_rates_by_script_run(
        jsonl_loc,
        session_run_ids,
        thresholds=(90.0, 45.0),
        session_id=current_session_id,
    )

    plot_success_rates_by_script_run(
        success_stats_by_script,
        title=f"Success by script_run_id (session {current_session_id})",
    )

    # Optimizer map (run_id -> optimizer) for legends
    optimizer_map = get_optimizer_map(jsonl_loc, session_run_ids, session_id=current_session_id)

    script_run_map = get_script_run_map(
        jsonl_loc,
        session_run_ids,
        session_id=current_session_id,
    )

    time_data = get_time_vs_angle(jsonl_loc, session_run_ids, session_id=current_session_id)
    plot_time_vs_angle(
        time_data,
        style_map=style_map,
        script_run_map=script_run_map,
        title=f"Speed vs Quality (session {current_session_id})",
    )

    # Get sparsity stats
    sparsity_stats = extract_sparsity(jsonl_loc, session_run_ids, session_id=current_session_id)

    # Optional: histogram (not style-dependent)
    # plot_sparsity_hist(sparsity_stats)

    # Scatter sparsity vs quality with shared style + optimizer legend
    plot_sparsity_vs_angle(
        jsonl_loc,
        session_run_ids,
        style_map=style_map,
        script_run_map=script_run_map,
        session_id=current_session_id,
        title=f"Sparsity vs Quality (session {current_session_id})",
    )

    # Time per run with shared colors
    # times_data = get_times(jsonl_loc, session_run_ids, session_id=current_session_id)
    # plot_times_bar(times_data, style_map=style_map,
    #                title=f"Time Per Run (session {current_session_id})")
    times_data = get_times(jsonl_loc, session_run_ids, session_id=current_session_id)
    plot_times_bar(
        times_data,
        style_map=style_map,
        optimizer_map=optimizer_map,
        script_run_map=script_run_map,
        title=f"Time Per Run (session {current_session_id})",
    )
