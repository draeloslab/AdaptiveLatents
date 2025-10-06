import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_stim_metrics(
    jsonl_path,
    row_start=None,
    row_end=None,
    alignment_metric="angle",   # "angle" → uses 'angle_deg'; "proj" → uses 'align_proj'
    save_path=None,
    dpi=120
):
    """
    Plot metrics logged in your JSONL.

    Parameters
    ----------
    jsonl_path : str or Path
        Path to the metrics JSONL file.
    row_start : int or None
        First row (0-based, inclusive) to plot. If None, start at 0.
    row_end : int or None
        Last row (0-based, exclusive) to plot. If None, go to end.
    alignment_metric : {"angle","proj"}
        Which alignment metric to plot in the middle panel.
        - "angle": uses 'angle_deg' (degrees; lower is better).
        - "proj" : uses 'align_proj' (unnormalized projection; higher is better).
    save_path : str or Path or None
        If provided, saves the figure to this path.
    dpi : int
        Figure DPI for saving/showing.
    """

    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No such file: {jsonl_path}")

    # Load rows
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines
                continue

    if not rows:
        raise ValueError("JSONL file has no valid rows.")

    # Slice by row indices
    n = len(rows)
    i0 = 0 if row_start is None else max(0, int(row_start))
    i1 = n if row_end is None else min(n, int(row_end))
    if i0 >= i1:
        raise ValueError(f"Invalid row range: start={i0}, end={i1}, total_rows={n}")
    data = rows[i0:i1]

    # Extract series with safe fallbacks
    def get_series(key, default=np.nan):
        vals = []
        for d in data:
            vals.append(d.get(key, default))
        return np.array(vals, dtype=float)

    eval_steps = get_series("eval")
    # If "eval" is missing, fall back to the slice index
    if np.isnan(eval_steps).all():
        eval_steps = np.arange(i0, i1)

    cos = get_series("align_cos")
    obj = get_series("obj")

    if alignment_metric == "angle":
        align = get_series("angle_deg")
        align_label = "Angle (deg, ↓ better)"
    elif alignment_metric == "proj":
        align = get_series("align_proj")
        align_label = "Alignment (projection, ↑ better)"
    else:
        raise ValueError("alignment_metric must be 'angle' or 'proj'")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    ax1, ax2, ax3 = axes

    ax1.plot(eval_steps, cos, marker="o", linewidth=1)
    ax1.set_ylabel("Cosine (↑ better)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(eval_steps, align, marker="o", linewidth=1)
    ax2.set_ylabel(align_label)
    ax2.grid(True, alpha=0.3)

    ax3.plot(eval_steps, obj, marker="o", linewidth=1)
    ax3.set_ylabel("Objective (↓ better)")
    ax3.set_xlabel("Evaluation step")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"Stim Design Metrics ({jsonl_path.name}) [{i0}:{i1}]", fontsize=12)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        print(f"Saved plot to: {save_path}")

    plt.show()

jsonl_loc = r"C:\Users\secom\OneDrive\Documents\DraelosLab\AdaptiveLatents\AdaptiveLatents\workspace\Alexworkspace\metrics_data.jsonl"


# choose rows to plot
plot_stim_metrics(jsonl_loc, row_start=576, row_end=605)

# # Use projection-based alignment on the middle plot
# plot_stim_metrics(jsonl_loc, alignment_metric="proj")

# # Save to PNG
# plot_stim_metrics(r"C:\...\metrics_data.jsonl", row_start=0, row_end=50,
#                   save_path=r"C:\...\stim_plot_0_50.png")
