import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe inside Docker / no display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# to call this script:
# python visualise.py --snapshotdate "2025-01-01"
#
# Reads all monitoring parquets from:
#   datamart/gold/model_monitoring/<modelname>/
# Saves charts to:
#   datamart/gold/model_monitoring/charts/


# ---------------------------------------------------------------------------
# PSI thresholds (industry standard)
# ---------------------------------------------------------------------------
PSI_STABLE  = 0.10   # below this  → stable
PSI_MONITOR = 0.20   # 0.10–0.20   → slight shift / monitor
                     # above 0.20  → significant drift / consider retrain

# Gini warning threshold — flag months below this for investigation
GINI_WARN = 0.20


def load_all_monitoring(monitoring_base_dir="datamart/gold/model_monitoring/"):
    """
    Scan all subdirectories of monitoring_base_dir for *.parquet files.
    Returns a combined pandas DataFrame, one row per (model, snapshot_month).
    """
    records = []
    model_dirs = [
        d for d in glob.glob(os.path.join(monitoring_base_dir, "*"))
        if os.path.isdir(d) and not d.endswith("charts")
    ]

    assert model_dirs, f"No model monitoring directories found under {monitoring_base_dir}"

    for model_dir in model_dirs:
        parquet_files = glob.glob(os.path.join(model_dir, "*.parquet"))
        if not parquet_files:
            continue
        for f in parquet_files:
            df = pd.read_parquet(f)
            records.append(df)

    combined = pd.concat(records, ignore_index=True)
    combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"])
    combined = combined.sort_values(["model_name", "snapshot_date"]).reset_index(drop=True)

    print(f"Loaded {len(combined)} monitoring records across {combined['model_name'].nunique()} model(s)")
    print(combined.groupby("model_name").size().to_string())
    return combined


def plot_gini_over_time(df, output_dir, snapshotdate_str):
    """
    Line chart: Gini over time for each model.

    - One line per model (champion vs challenger, different colours)
    - Shaded warning band below GINI_WARN
    - Training period (Jan 2023–Jun 2024) shaded differently from monitoring period
    - Markers at each data point
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colours = {0: "#1C7293", 1: "#F96167", 2: "#02C39A", 3: "#F9E795"}
    model_names = sorted(df["model_name"].unique())

    for i, mname in enumerate(model_names):
        mdf = df[df["model_name"] == mname].copy()
        gini_valid = mdf.dropna(subset=["gini"])

        colour = colours.get(i, "#888888")
        short_name = mname.split("_model_")[-1] if "_model_" in mname else mname

        ax.plot(
            gini_valid["snapshot_date"],
            gini_valid["gini"],
            marker="o",
            linewidth=2,
            color=colour,
            label=short_name,
            zorder=3,
        )
        # mark months where gini is None (no label data)
        gini_null = mdf[mdf["gini"].isna()]
        if len(gini_null) > 0:
            ax.scatter(
                gini_null["snapshot_date"],
                [0.0] * len(gini_null),
                marker="x",
                color=colour,
                s=60,
                zorder=4,
                label=f"{short_name} (no labels)",
            )

    # warning band: gini below GINI_WARN
    ax.axhspan(0, GINI_WARN, color="#FFE0E0", alpha=0.5, zorder=0, label=f"Gini < {GINI_WARN} (low — investigate)")
    ax.axhline(GINI_WARN, color="#CC3333", linewidth=1.2, linestyle="--", zorder=2)

    # vertical line separating training period from post-deployment
    train_cutoff = pd.Timestamp("2024-07-01")   # Jul 2024 = OOT start
    deploy_line  = pd.Timestamp("2024-09-01")   # Sep 2024 = deployment
    ax.axvline(deploy_line, color="#444444", linewidth=1.5, linestyle=":", zorder=2)
    ax.text(deploy_line, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
            " ← Deployment", fontsize=9, color="#444444", va="top")

    ax.set_title("Model Gini Over Time (Jan 2023 – Jan 2025)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Snapshot Date", fontsize=11)
    ax.set_ylabel("Gini Coefficient", fontsize=11)
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"gini_over_time_{snapshotdate_str.replace('-','_')}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_psi_over_time(df, output_dir, snapshotdate_str):
    """
    Bar chart: PSI over time for each model.

    - One bar group per month (one bar per model, side-by-side)
    - Horizontal reference lines at 0.10 (monitor) and 0.20 (retrain)
    - Bars colour-coded: green < 0.10, amber 0.10–0.20, red > 0.20
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    model_names = sorted(df["model_name"].unique())
    n_models    = len(model_names)
    all_dates   = sorted(pd.Timestamp(d) for d in df["snapshot_date"].unique())
    x           = np.arange(len(all_dates))
    bar_width   = 0.8 / max(n_models, 1)

    model_colours = {0: "#1C7293", 1: "#F96167", 2: "#02C39A", 3: "#F9E795"}

    for i, mname in enumerate(model_names):
        mdf    = df[df["model_name"] == mname].set_index("snapshot_date")
        psi_vals = []
        bar_cols = []

        for d in all_dates:
            val = mdf.loc[d, "psi"] if d in mdf.index else None
            psi_vals.append(val if pd.notna(val) else 0)
            if val is None or pd.isna(val):
                bar_cols.append("#CCCCCC")
            elif val < PSI_STABLE:
                bar_cols.append("#2ECC71")   # green  — stable
            elif val < PSI_MONITOR:
                bar_cols.append("#F39C12")   # amber  — monitor
            else:
                bar_cols.append("#E74C3C")   # red    — drift

        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            psi_vals,
            width=bar_width,
            color=bar_cols,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        # invisible line for legend
        short = mname.split("_model_")[-1] if "_model_" in mname else mname
        ax.plot([], [], color=model_colours.get(i, "#888888"),
                linewidth=6, label=short)

    # reference lines
    ax.axhline(PSI_STABLE,  color="#27AE60", linewidth=1.5, linestyle="--",
               label=f"PSI = {PSI_STABLE} (stable threshold)")
    ax.axhline(PSI_MONITOR, color="#E74C3C", linewidth=1.5, linestyle="--",
               label=f"PSI = {PSI_MONITOR} (retrain threshold)")

    # vertical deployment marker
    deploy_idx = [i for i, d in enumerate(all_dates) if d == pd.Timestamp("2024-09-01")]
    if deploy_idx:
        ax.axvline(deploy_idx[0], color="#444444", linewidth=1.5,
                   linestyle=":", zorder=2, label="Deployment")

    # legend for bar colours
    stable_patch  = mpatches.Patch(color="#2ECC71", label=f"PSI < {PSI_STABLE}  (stable)")
    monitor_patch = mpatches.Patch(color="#F39C12", label=f"PSI {PSI_STABLE}–{PSI_MONITOR} (monitor)")
    drift_patch   = mpatches.Patch(color="#E74C3C", label=f"PSI > {PSI_MONITOR} (drift)")
    na_patch      = mpatches.Patch(color="#CCCCCC", label="No data")

    ax.set_title("Population Stability Index (PSI) Over Time", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Snapshot Date", fontsize=11)
    ax.set_ylabel("PSI", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in all_dates],
        rotation=45, ha="right", fontsize=8
    )
    ax.legend(
        handles=[stable_patch, monitor_patch, drift_patch, na_patch],
        loc="upper right", fontsize=9, framealpha=0.9
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"psi_over_time_{snapshotdate_str.replace('-','_')}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_mean_score_over_time(df, output_dir, snapshotdate_str):
    """
    Line chart: mean prediction score + % high-risk over time.
    Secondary axis for pct_high_risk.
    """
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    colours  = {0: "#1C7293", 1: "#F96167"}
    model_names = sorted(df["model_name"].unique())

    for i, mname in enumerate(model_names):
        mdf    = df[df["model_name"] == mname].dropna(subset=["mean_score"])
        colour = colours.get(i, "#888888")
        short  = mname.split("_model_")[-1] if "_model_" in mname else mname

        ax1.plot(mdf["snapshot_date"], mdf["mean_score"],
                 marker="o", linewidth=2, color=colour, label=f"{short} mean score")
        ax2.plot(mdf["snapshot_date"], mdf["pct_high_risk"],
                 marker="s", linewidth=1.5, linestyle="--", color=colour, alpha=0.6,
                 label=f"{short} % high-risk")

    ax1.set_title("Mean Score & % High-Risk Over Time", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Snapshot Date", fontsize=11)
    ax1.set_ylabel("Mean Prediction Score (0–1)", fontsize=11, color="#1C7293")
    ax2.set_ylabel("% Customers Predicted High-Risk", fontsize=11, color="#888888")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_facecolor("#FAFAFA")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()

    out_path = os.path.join(output_dir, f"score_distribution_{snapshotdate_str.replace('-','_')}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    monitoring_base  = "datamart/gold/model_monitoring/"
    charts_dir       = os.path.join(monitoring_base, "charts/")

    # -----------------------------------------------------------------------
    # Load all monitoring data
    # -----------------------------------------------------------------------
    # Guard: no monitoring data yet (model not trained) — exit cleanly
    model_dirs = [
        d for d in glob.glob(os.path.join(monitoring_base, "*"))
        if os.path.isdir(d) and not d.endswith("charts")
    ]
    parquet_files = []
    for d in model_dirs:
        parquet_files.extend(glob.glob(os.path.join(d, "*.parquet")))

    if not parquet_files:
        print("No monitoring data found yet — model not yet trained. Skipping visualisation.")
        print('\n\n---completed job (no-op)---\n\n')
        return

    df = load_all_monitoring(monitoring_base)

    print(f"\nSnapshot date range: {df['snapshot_date'].min().date()} → {df['snapshot_date'].max().date()}")
    print(f"\nSample rows:")
    print(df[["snapshot_date", "model_name", "gini", "psi", "mean_score", "pct_high_risk"]].head(10).to_string(index=False))

    # -----------------------------------------------------------------------
    # Chart 1: Gini over time
    # -----------------------------------------------------------------------
    gini_path = plot_gini_over_time(df, charts_dir, snapshotdate)

    # -----------------------------------------------------------------------
    # Chart 2: PSI over time
    # -----------------------------------------------------------------------
    psi_path = plot_psi_over_time(df, charts_dir, snapshotdate)

    # -----------------------------------------------------------------------
    # Chart 3: Mean score + % high-risk over time
    # -----------------------------------------------------------------------
    score_path = plot_mean_score_over_time(df, charts_dir, snapshotdate)

    # -----------------------------------------------------------------------
    # Summary stats table
    # -----------------------------------------------------------------------
    print("\n=== Monitoring Summary (all months) ===")
    summary = df.groupby("model_name").agg(
        months=("snapshot_date", "count"),
        gini_mean=("gini", "mean"),
        gini_min=("gini", "min"),
        gini_max=("gini", "max"),
        psi_mean=("psi",  "mean"),
        psi_max=("psi",   "max"),
        pct_high_risk_mean=("pct_high_risk", "mean"),
    ).round(3)
    print(summary.to_string())

    # flag months with Gini < GINI_WARN
    low_gini = df[df["gini"] < GINI_WARN].dropna(subset=["gini"])
    if len(low_gini) > 0:
        print(f"\n⚠  Months with Gini < {GINI_WARN} ({len(low_gini)} records):")
        print(low_gini[["snapshot_date","model_name","gini"]].to_string(index=False))

    # flag months with PSI > PSI_MONITOR
    high_psi = df[df["psi"] > PSI_MONITOR].dropna(subset=["psi"])
    if len(high_psi) > 0:
        print(f"\n⚠  Months with PSI > {PSI_MONITOR} ({len(high_psi)} records):")
        print(high_psi[["snapshot_date","model_name","psi"]].to_string(index=False))

    print(f"\nCharts saved to: {charts_dir}")
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise model monitoring results")
    parser.add_argument("--snapshotdate", type=str, required=True,
                        help="Run date YYYY-MM-DD (used to label output files)")
    args = parser.parse_args()
    main(args.snapshotdate)
