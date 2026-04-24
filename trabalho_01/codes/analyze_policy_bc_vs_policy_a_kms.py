"""Analyze Policy B and Policy C Monte Carlo results against baseline Policy A.

This script is the post-processing entrypoint for the per-regime Monte Carlo
study. It does not run the Policy B or Policy C simulations itself. Instead,
it reads their saved repetition files, reconstructs the exact same evaluation
splits, computes the baseline Policy A on those matched rows, and then writes
comparison tables and plots for each regime.

What this script does:
- Read `run_config.json` from the Policy B and Policy C run directories.
- Verify that both simulation runs were generated with compatible settings.
- For each KMS regime:
  - load the original regime matrix,
  - find the repetitions that exist in both Policy B and Policy C outputs,
  - rebuild the matching evaluation split for each repetition,
  - compute Policy A on that exact evaluation split,
  - combine Policy A, Policy B, and Policy C into one comparison table,
  - summarize mean performance and uncertainty by observation budget,
  - generate regime-level plots and a text summary.
- After processing all regimes, create cross-regime CSV summaries and a heatmap
  showing the best policy by regime and budget.

Step-by-step structure of the file:
1. Import block:
   The script imports NumPy and Matplotlib for aggregation and plotting, then
   imports the shared helper module so the reconstructed Monte Carlo splits are
   identical to those used by the simulation scripts.
2. Analysis configuration section:
   This is where you point the script at the Policy B and Policy C run
   directories and decide whether to force rebuilding existing analysis files.
3. Validation helpers:
   `validate_policy_run_configs()` ensures the two simulation runs are
   directly comparable. The analysis stops if the dataset path, threshold,
   budgets, number of repetitions, or other critical settings differ.
4. Policy A reconstruction helpers:
   `build_policy_a_row()` rebuilds the baseline on the exact same evaluation
   rows used by Policy B and Policy C. This keeps the comparison fair.
5. Aggregation helpers:
   `normalize_policy_row()` and `summarize_comparison_rows()` convert raw
   repetition files into comparison tables with means, standard deviations,
   pooled Wilson intervals, and gains relative to Policy A.
6. Plotting and text-summary helpers:
   The plotting functions generate regime-specific figures and one cross-regime
   heatmap, while the summary writers generate concise human-readable files.
7. Main analysis runner:
   `run_configured_policy_analysis()` orchestrates the whole analysis
   pipeline, supports resumable reruns, and writes the final combined outputs.

How restart/resume works:
- If `combined_policy_rows.csv` and `summary_by_budget.csv` already exist for a
  regime, that regime is skipped when `FORCE_REBUILD_ANALYSIS` is `False`.
- Existing summary CSVs are loaded back into memory so the cross-regime files
  can still be rebuilt without recomputing the skipped regimes.
- This keeps the analysis resumable even when only part of the 14-regime study
  was processed before an interruption.

Output layout:
- `<analysis>/<regime>/combined_policy_rows.csv`: repetition-level comparison
  rows for Policy A, Policy B, and Policy C.
- `<analysis>/<regime>/summary_by_budget.csv`: aggregated metrics per budget.
- `<analysis>/<regime>/op_vs_budget.png`: mean outage with uncertainty bars.
- `<analysis>/<regime>/gain_vs_policy_a.png`: mean gain of Policies B and C
  relative to Policy A.
- `<analysis>/<regime>/summary.txt`: compact regime-level interpretation.
- `<analysis>/cross_regime_summary.csv`: merged summary across all processed
  regimes.
- `<analysis>/best_policy_by_regime_and_budget.csv`: best mean policy for each
  regime/budget pair.
- `<analysis>/best_policy_heatmap.png`: visual summary of the winning policy.
- `<analysis>/summary.txt`: compact cross-regime text summary.

Example command:
```bash
python src/architectures/dataset_analysis/analyze_policy_bc_vs_policy_a_kms.py
```

Typical workflow:
1. Run `policy_b_kms_per_regime_mc.py`.
2. Run `policy_c_kms_per_regime_mc.py`.
3. Run this analysis script to compare both policies against baseline
   Policy A and generate the graphs.
"""

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = REPO_ROOT / "src"
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from policy_mc_kms_common import (
    aggregate_result_rows_from_run,
    build_policy_a_indices,
    build_rep_rng,
    compute_wilson_interval,
    ensure_directory,
    ensure_matching_run_config,
    evaluate_ideal_ports,
    evaluate_selected_ports,
    format_indices_one_based,
    KMS_REGIME_LABELS,
    list_completed_repetitions,
    load_kms_regime,
    log_status,
    read_csv_rows,
    read_json_file,
    split_design_eval_indices,
    threshold_to_binary,
    write_csv_rows_atomic,
    write_json_atomic,
)


# -----------------------------------------------------------------------------
# Analysis configuration
# -----------------------------------------------------------------------------
RUNS_ROOT = REPO_ROOT / "runs" / "dataset_analysis"
POLICY_B_RUN_DIR = RUNS_ROOT / "policy_b_kms_per_regime_mc"
POLICY_C_RUN_DIR = RUNS_ROOT / "policy_c_kms_per_regime_mc"
ANALYSIS_RUN_NAME = "analysis_policy_bc_vs_policy_a_kms"
ANALYSIS_OUTPUT_DIR = RUNS_ROOT / ANALYSIS_RUN_NAME
FORCE_REBUILD_ANALYSIS = False
ALLOW_PARTIAL_POLICY_RUNS = False


COMPARISON_FIELDNAMES: tuple[str, ...] = (
    "dataset",
    "regime_index",
    "rep",
    "policy",
    "n_ports",
    "selection_order_1based",
    "sorted_pattern_1based",
    "observed_op",
    "ideal_op",
    "gain_vs_policy_a",
    "gap_to_ideal",
    "n_eval",
    "n_outage",
    "n_ideal_outage",
    "wilson_low",
    "wilson_high",
)
SUMMARY_FIELDNAMES: tuple[str, ...] = (
    "dataset",
    "regime_index",
    "policy",
    "n_ports",
    "n_reps",
    "mean_observed_op",
    "std_observed_op",
    "ci95_half_width",
    "pooled_observed_op",
    "pooled_wilson_low",
    "pooled_wilson_high",
    "mean_ideal_op",
    "mean_gain_vs_policy_a",
    "mean_gap_to_ideal",
)
BEST_POLICY_FIELDNAMES: tuple[str, ...] = (
    "dataset",
    "regime_index",
    "n_ports",
    "best_policy",
    "best_mean_observed_op",
)


def validate_policy_run_configs(
    policy_b_config: dict[str, Any],
    policy_c_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate that Policy B and Policy C runs can be compared safely.

    Args:
        policy_b_config (dict[str, Any]): Configuration loaded from the Policy B
            run directory.
        policy_c_config (dict[str, Any]): Configuration loaded from the Policy C
            run directory.

    Returns:
        dict[str, Any]: Shared comparison settings extracted from the two run
            configurations after compatibility checks pass.

    Raises:
        ValueError: If any comparison-critical configuration entry differs.
    """
    keys_to_match = (
        "common_dataset_path",
        "data_seed",
        "threshold",
        "snr_linear",
        "total_ports",
        "target_ports",
        "num_mc_reps",
        "mc_eval_fraction",
        "regime_labels",
    )
    for key in keys_to_match:
        if policy_b_config.get(key) != policy_c_config.get(key):
            raise ValueError(f"Policy B and Policy C configurations differ for key '{key}'.")

    return {
        "common_dataset_path": str(policy_b_config["common_dataset_path"]),
        "data_seed": int(policy_b_config["data_seed"]),
        "threshold": float(policy_b_config["threshold"]),
        "snr_linear": float(policy_b_config["snr_linear"]),
        "total_ports": int(policy_b_config["total_ports"]),
        "target_ports": tuple(int(value) for value in policy_b_config["target_ports"]),
        "num_mc_reps": int(policy_b_config["num_mc_reps"]),
        "mc_eval_fraction": float(policy_b_config["mc_eval_fraction"]),
        "regime_labels": tuple(str(value) for value in policy_b_config["regime_labels"]),
    }


def build_analysis_run_config(shared_config: dict[str, Any]) -> dict[str, Any]:
    """Build the persisted configuration for the analysis run directory.

    Args:
        shared_config (dict[str, Any]): Shared validated settings from the
            Policy B and Policy C simulation runs.

    Returns:
        dict[str, Any]: JSON-serializable analysis configuration dictionary.
    """
    return {
        "script_name": Path(__file__).stem,
        "policy_b_run_dir": str(POLICY_B_RUN_DIR),
        "policy_c_run_dir": str(POLICY_C_RUN_DIR),
        "analysis_output_dir": str(ANALYSIS_OUTPUT_DIR),
        "force_rebuild_analysis": bool(FORCE_REBUILD_ANALYSIS),
        "allow_partial_policy_runs": bool(ALLOW_PARTIAL_POLICY_RUNS),
        **shared_config,
    }


def index_policy_rows_by_rep_and_budget(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, str]]:
    """Index policy result rows by repetition and budget.

    Args:
        rows (list[dict[str, str]]): CSV rows loaded from one policy run for a
            single regime.

    Returns:
        dict[tuple[int, int], dict[str, str]]: Mapping keyed by
            ``(rep_index, n_ports)``.

    Raises:
        ValueError: If duplicate entries are found for the same key.
    """
    index: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        key = (int(row["rep"]), int(row["n_ports"]))
        if key in index:
            raise ValueError(f"Duplicate policy row detected for key {key}.")
        index[key] = row
    return index


def build_policy_a_row(
    *,
    regime_label: str,
    regime_index: int,
    rep_index: int,
    n_ports: int,
    binary_eval: np.ndarray,
    total_ports: int,
    ideal_op: float,
    n_ideal_outage: int,
) -> dict[str, Any]:
    """Compute one Policy A comparison row for a given repetition and budget.

    Args:
        regime_label (str): Human-readable regime label.
        regime_index (int): Zero-based regime index.
        rep_index (int): Zero-based repetition index.
        n_ports (int): Observation budget for Policy A.
        binary_eval (np.ndarray): Evaluation exceedance matrix for the current
            regime and repetition.
        total_ports (int): Total number of candidate ports in the regime.
        ideal_op (float): Ideal outage probability already computed on the same
            evaluation rows. Passing this value avoids recomputing the full-port
            ideal baseline for every budget.
        n_ideal_outage (int): Number of ideal-outage rows on the same
            evaluation split.

    Returns:
        dict[str, Any]: Comparison row aligned with ``COMPARISON_FIELDNAMES``.

    Raises:
        ValueError: If Policy A cannot be built for the requested budget.
    """
    selected_indices = build_policy_a_indices(total_ports=total_ports, n_ports=n_ports)
    observed_op, n_eval, n_outage = evaluate_selected_ports(binary_eval=binary_eval, selected_indices=selected_indices)
    wilson_low, wilson_high = compute_wilson_interval(n_outage=n_outage, n_total=n_eval)
    selection_text = format_indices_one_based(selected_indices)
    return {
        "dataset": regime_label,
        "regime_index": int(regime_index),
        "rep": int(rep_index),
        "policy": "policy_a",
        "n_ports": int(n_ports),
        "selection_order_1based": selection_text,
        "sorted_pattern_1based": selection_text,
        "observed_op": float(observed_op),
        "ideal_op": float(ideal_op),
        "gain_vs_policy_a": 0.0,
        "gap_to_ideal": float(observed_op - ideal_op),
        "n_eval": int(n_eval),
        "n_outage": int(n_outage),
        "n_ideal_outage": int(n_ideal_outage),
        "wilson_low": float(wilson_low),
        "wilson_high": float(wilson_high),
    }


def normalize_policy_row(
    source_row: dict[str, str],
    *,
    gain_vs_policy_a: float,
) -> dict[str, Any]:
    """Convert one stored Policy B or Policy C CSV row into comparison format.

    Args:
        source_row (dict[str, str]): Raw row loaded from a Policy B or Policy C
            repetition file.
        gain_vs_policy_a (float): Improvement relative to Policy A on the same
            regime, repetition, and budget. Positive values mean the policy
            achieved lower outage than Policy A.

    Returns:
        dict[str, Any]: Comparison row aligned with ``COMPARISON_FIELDNAMES``.
    """
    observed_op = float(source_row["observed_op"])
    ideal_op = float(source_row["ideal_op"])
    return {
        "dataset": str(source_row["dataset"]),
        "regime_index": int(source_row["regime_index"]),
        "rep": int(source_row["rep"]),
        "policy": str(source_row["policy"]),
        "n_ports": int(source_row["n_ports"]),
        "selection_order_1based": str(source_row["selection_order_1based"]),
        "sorted_pattern_1based": str(source_row["sorted_pattern_1based"]),
        "observed_op": observed_op,
        "ideal_op": ideal_op,
        "gain_vs_policy_a": float(gain_vs_policy_a),
        "gap_to_ideal": float(observed_op - ideal_op),
        "n_eval": int(source_row["n_eval"]),
        "n_outage": int(source_row["n_outage"]),
        "n_ideal_outage": int(source_row["n_ideal_outage"]),
        "wilson_low": float(source_row["wilson_low"]),
        "wilson_high": float(source_row["wilson_high"]),
    }


def summarize_comparison_rows(
    comparison_rows: list[dict[str, Any]],
    regime_label: str,
    regime_index: int,
) -> list[dict[str, Any]]:
    """Aggregate repetition-level comparison rows into one summary per budget.

    Args:
        comparison_rows (list[dict[str, Any]]): Repetition-level rows for a
            single regime.
        regime_label (str): Human-readable regime label used in the summary.
        regime_index (int): Zero-based regime index.

    Returns:
        list[dict[str, Any]]: Summary rows aligned with
            ``SUMMARY_FIELDNAMES``.

    Raises:
        ValueError: If ``comparison_rows`` is empty.
    """
    if len(comparison_rows) == 0:
        raise ValueError("comparison_rows must not be empty.")

    grouped_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in comparison_rows:
        key = (str(row["policy"]), int(row["n_ports"]))
        grouped_rows.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (policy_name, n_ports), rows in sorted(grouped_rows.items(), key=lambda item: (item[0][0], item[0][1])):
        observed_values = np.array([float(row["observed_op"]) for row in rows], dtype=np.float64)
        ideal_values = np.array([float(row["ideal_op"]) for row in rows], dtype=np.float64)
        gains = np.array([float(row["gain_vs_policy_a"]) for row in rows], dtype=np.float64)
        gap_values = np.array([float(row["gap_to_ideal"]) for row in rows], dtype=np.float64)
        outage_counts = np.array([int(row["n_outage"]) for row in rows], dtype=np.int64)
        eval_counts = np.array([int(row["n_eval"]) for row in rows], dtype=np.int64)
        pooled_n_outage = int(np.sum(outage_counts))
        pooled_n_eval = int(np.sum(eval_counts))
        pooled_wilson_low, pooled_wilson_high = compute_wilson_interval(
            n_outage=pooled_n_outage,
            n_total=pooled_n_eval,
        )

        # The repetition-to-repetition spread quantifies Monte Carlo stability
        # of the learned policies, while the pooled Wilson interval reflects
        # Bernoulli uncertainty after combining all evaluation rows.
        std_observed = float(np.std(observed_values, ddof=1)) if len(observed_values) > 1 else 0.0
        ci95_half_width = float(1.96 * std_observed / math.sqrt(len(observed_values)))
        summary_rows.append(
            {
                "dataset": regime_label,
                "regime_index": int(regime_index),
                "policy": policy_name,
                "n_ports": int(n_ports),
                "n_reps": int(len(rows)),
                "mean_observed_op": float(np.mean(observed_values)),
                "std_observed_op": std_observed,
                "ci95_half_width": ci95_half_width,
                "pooled_observed_op": float(pooled_n_outage / pooled_n_eval),
                "pooled_wilson_low": float(pooled_wilson_low),
                "pooled_wilson_high": float(pooled_wilson_high),
                "mean_ideal_op": float(np.mean(ideal_values)),
                "mean_gain_vs_policy_a": float(np.mean(gains)),
                "mean_gap_to_ideal": float(np.mean(gap_values)),
            }
        )
    return summary_rows


def write_regime_summary_text(
    regime_output_dir: Path,
    regime_label: str,
    summary_rows: list[dict[str, Any]],
) -> Path:
    """Write a concise human-readable summary for one regime analysis.

    Args:
        regime_output_dir (Path): Output directory for the analyzed regime.
        regime_label (str): Human-readable regime label.
        summary_rows (list[dict[str, Any]]): Aggregated summary rows for the
            regime.

    Returns:
        Path: Path to the written summary text file.

    Raises:
        OSError: If the file cannot be written.
    """
    best_by_budget: dict[int, tuple[str, float]] = {}
    for row in summary_rows:
        n_ports = int(row["n_ports"])
        candidate = (str(row["policy"]), float(row["mean_observed_op"]))
        if n_ports not in best_by_budget or candidate[1] < best_by_budget[n_ports][1]:
            best_by_budget[n_ports] = candidate

    lines = [f"Regime: {regime_label}", ""]
    lines.append("Best policy by budget:")
    for n_ports in sorted(best_by_budget):
        policy_name, mean_observed_op = best_by_budget[n_ports]
        lines.append(f"- N={n_ports}: {policy_name} with mean observed OP={mean_observed_op:.6f}")
    output_path = regime_output_dir / "summary.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def plot_regime_op_vs_budget(
    regime_output_dir: Path,
    regime_label: str,
    summary_rows: list[dict[str, Any]],
) -> Path:
    """Plot mean outage probability versus budget for one regime.

    Args:
        regime_output_dir (Path): Output directory for the regime plots.
        regime_label (str): Human-readable regime label.
        summary_rows (list[dict[str, Any]]): Aggregated summary rows for the
            regime.

    Returns:
        Path: Saved PNG path.

    Raises:
        OSError: If the figure cannot be written.
    """
    policy_styles = {
        "policy_a": {"label": "Policy A", "color": "tab:blue", "marker": "o"},
        "policy_b": {"label": "Policy B", "color": "tab:orange", "marker": "s"},
        "policy_c": {"label": "Policy C", "color": "tab:green", "marker": "^"},
    }
    summary_by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        summary_by_policy.setdefault(str(row["policy"]), []).append(row)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for policy_name, style in policy_styles.items():
        if policy_name not in summary_by_policy:
            continue
        ordered_rows = sorted(summary_by_policy[policy_name], key=lambda item: int(item["n_ports"]))
        x_values = np.array([int(row["n_ports"]) for row in ordered_rows], dtype=np.int64)
        y_values = np.array([float(row["mean_observed_op"]) for row in ordered_rows], dtype=np.float64)
        y_errors = np.array([float(row["ci95_half_width"]) for row in ordered_rows], dtype=np.float64)
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            capsize=4,
            label=style["label"],
        )

    if "policy_a" in summary_by_policy:
        ideal_rows = sorted(summary_by_policy["policy_a"], key=lambda item: int(item["n_ports"]))
        ax.plot(
            [int(row["n_ports"]) for row in ideal_rows],
            [float(row["mean_ideal_op"]) for row in ideal_rows],
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Ideal OP",
        )

    ax.set_title(f"Observed-only outage by budget: {regime_label}")
    ax.set_xlabel("Observed ports")
    ax.set_ylabel("Outage probability")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    output_path = regime_output_dir / "op_vs_budget.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_regime_gain_vs_policy_a(
    regime_output_dir: Path,
    regime_label: str,
    summary_rows: list[dict[str, Any]],
) -> Path:
    """Plot mean gain relative to Policy A for Policy B and Policy C.

    Args:
        regime_output_dir (Path): Output directory for the regime plots.
        regime_label (str): Human-readable regime label.
        summary_rows (list[dict[str, Any]]): Aggregated summary rows for the
            regime.

    Returns:
        Path: Saved PNG path.

    Raises:
        OSError: If the figure cannot be written.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    policy_styles = {
        "policy_b": {"label": "Policy B gain", "color": "tab:orange", "marker": "s"},
        "policy_c": {"label": "Policy C gain", "color": "tab:green", "marker": "^"},
    }
    for policy_name, style in policy_styles.items():
        ordered_rows = sorted(
            (row for row in summary_rows if str(row["policy"]) == policy_name),
            key=lambda item: int(item["n_ports"]),
        )
        if len(ordered_rows) == 0:
            continue
        x_values = np.array([int(row["n_ports"]) for row in ordered_rows], dtype=np.int64)
        y_values = np.array([float(row["mean_gain_vs_policy_a"]) for row in ordered_rows], dtype=np.float64)
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            label=style["label"],
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title(f"Gain relative to Policy A: {regime_label}")
    ax.set_xlabel("Observed ports")
    ax.set_ylabel("Mean gain vs Policy A")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    output_path = regime_output_dir / "gain_vs_policy_a.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_best_policy_heatmap(
    output_dir: Path,
    cross_regime_summary_rows: list[dict[str, Any]],
    target_ports: tuple[int, ...],
) -> Path:
    """Plot the best policy by regime and observation budget.

    Args:
        output_dir (Path): Root analysis output directory.
        cross_regime_summary_rows (list[dict[str, Any]]): Summary rows from all
            analyzed regimes.
        target_ports (tuple[int, ...]): Observation budgets included in the
            study.

    Returns:
        Path: Saved PNG path.

    Raises:
        OSError: If the figure cannot be written.
    """
    regime_count = len(KMS_REGIME_LABELS)
    budget_count = len(target_ports)
    matrix = np.full((regime_count, budget_count), np.nan, dtype=np.float64)
    policy_to_value = {"policy_a": 0.0, "policy_b": 1.0, "policy_c": 2.0}

    for regime_index, regime_label in enumerate(KMS_REGIME_LABELS):
        for budget_index, n_ports in enumerate(target_ports):
            candidate_rows = [
                row
                for row in cross_regime_summary_rows
                if str(row["dataset"]) == regime_label and int(row["n_ports"]) == n_ports
            ]
            if len(candidate_rows) == 0:
                continue
            best_row = min(candidate_rows, key=lambda item: float(item["mean_observed_op"]))
            matrix[regime_index, budget_index] = policy_to_value[str(best_row["policy"])]

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    cmap = ListedColormap(["tab:blue", "tab:orange", "tab:green"])
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=2.0)
    ax.set_title("Best policy by regime and budget")
    ax.set_xlabel("Observed ports")
    ax.set_ylabel("Regime")
    ax.set_xticks(np.arange(budget_count))
    ax.set_xticklabels([str(value) for value in target_ports])
    ax.set_yticks(np.arange(regime_count))
    ax.set_yticklabels(list(KMS_REGIME_LABELS))
    colorbar = fig.colorbar(image, ax=ax, ticks=[0.0, 1.0, 2.0])
    colorbar.ax.set_yticklabels(["Policy A", "Policy B", "Policy C"])
    fig.tight_layout()
    output_path = output_dir / "best_policy_heatmap.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def write_cross_regime_summary_text(
    output_dir: Path,
    cross_regime_summary_rows: list[dict[str, Any]],
    skipped_regimes: list[str],
    target_ports: tuple[int, ...],
) -> Path:
    """Write a compact text summary across all analyzed regimes.

    Args:
        output_dir (Path): Root analysis output directory.
        cross_regime_summary_rows (list[dict[str, Any]]): Combined summary rows
            across all processed regimes.
        skipped_regimes (list[str]): Regimes that were skipped because source
            policy runs were incomplete.
        target_ports (tuple[int, ...]): Observation budgets included in the
            study.

    Returns:
        Path: Saved summary text path.

    Raises:
        OSError: If the file cannot be written.
    """
    lines = ["Cross-regime summary", ""]
    for n_ports in target_ports:
        candidate_rows = [row for row in cross_regime_summary_rows if int(row["n_ports"]) == n_ports]
        if len(candidate_rows) == 0:
            continue
        mean_by_policy: dict[str, float] = {}
        for policy_name in ("policy_a", "policy_b", "policy_c"):
            policy_rows = [row for row in candidate_rows if str(row["policy"]) == policy_name]
            if len(policy_rows) == 0:
                continue
            mean_by_policy[policy_name] = float(
                np.mean([float(row["mean_observed_op"]) for row in policy_rows], dtype=np.float64)
            )
        if len(mean_by_policy) == 0:
            continue
        best_policy = min(mean_by_policy, key=mean_by_policy.get)
        lines.append(f"- N={n_ports}: best mean policy across processed regimes is {best_policy}")
        for policy_name in ("policy_a", "policy_b", "policy_c"):
            if policy_name in mean_by_policy:
                lines.append(f"  {policy_name}: {mean_by_policy[policy_name]:.6f}")
        lines.append("")

    if skipped_regimes:
        lines.append("Skipped regimes:")
        for regime_label in skipped_regimes:
            lines.append(f"- {regime_label}")
    output_path = output_dir / "summary.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_configured_policy_analysis() -> dict[str, Any]:
    """Run the baseline comparison analysis for Policies A, B, and C.

    Args:
        None: The analysis reads all settings from the module-level constants.

    Returns:
        dict[str, Any]: Summary with the output directory and the processed
            regime list.

    Raises:
        FileNotFoundError: If the required Policy B or Policy C run
            directories are missing.
        ValueError: If the Policy B and Policy C run configurations are
            incompatible.
        OSError: If output artifacts cannot be written.
    """
    script_name = Path(__file__).stem
    if not POLICY_B_RUN_DIR.exists():
        raise FileNotFoundError(f"Policy B run directory not found: {POLICY_B_RUN_DIR}")
    if not POLICY_C_RUN_DIR.exists():
        raise FileNotFoundError(f"Policy C run directory not found: {POLICY_C_RUN_DIR}")

    policy_b_config = read_json_file(POLICY_B_RUN_DIR / "run_config.json")
    policy_c_config = read_json_file(POLICY_C_RUN_DIR / "run_config.json")
    shared_config = validate_policy_run_configs(policy_b_config=policy_b_config, policy_c_config=policy_c_config)

    analysis_output_dir = ensure_directory(ANALYSIS_OUTPUT_DIR)
    ensure_matching_run_config(analysis_output_dir, build_analysis_run_config(shared_config))

    cross_regime_summary_rows: list[dict[str, Any]] = []
    best_policy_rows: list[dict[str, Any]] = []
    processed_regimes: list[str] = []
    skipped_regimes: list[str] = []

    for regime_index, regime_label in enumerate(shared_config["regime_labels"]):
        regime_output_dir = ensure_directory(analysis_output_dir / regime_label)
        combined_csv_path = regime_output_dir / "combined_policy_rows.csv"
        summary_csv_path = regime_output_dir / "summary_by_budget.csv"

        if not FORCE_REBUILD_ANALYSIS and combined_csv_path.exists() and summary_csv_path.exists():
            log_status(script_name, "resume", f"Skipping {regime_label}: analysis files already exist")
            existing_summary_rows = []
            for row in read_csv_rows(summary_csv_path):
                existing_summary_rows.append(
                    {
                        "dataset": str(row["dataset"]),
                        "regime_index": int(row["regime_index"]),
                        "policy": str(row["policy"]),
                        "n_ports": int(row["n_ports"]),
                        "n_reps": int(row["n_reps"]),
                        "mean_observed_op": float(row["mean_observed_op"]),
                        "std_observed_op": float(row["std_observed_op"]),
                        "ci95_half_width": float(row["ci95_half_width"]),
                        "pooled_observed_op": float(row["pooled_observed_op"]),
                        "pooled_wilson_low": float(row["pooled_wilson_low"]),
                        "pooled_wilson_high": float(row["pooled_wilson_high"]),
                        "mean_ideal_op": float(row["mean_ideal_op"]),
                        "mean_gain_vs_policy_a": float(row["mean_gain_vs_policy_a"]),
                        "mean_gap_to_ideal": float(row["mean_gap_to_ideal"]),
                    }
                )
            processed_regimes.append(regime_label)
            cross_regime_summary_rows.extend(existing_summary_rows)
            for n_ports in shared_config["target_ports"]:
                candidate_rows = [row for row in existing_summary_rows if int(row["n_ports"]) == int(n_ports)]
                if len(candidate_rows) == 0:
                    continue
                best_row = min(candidate_rows, key=lambda item: float(item["mean_observed_op"]))
                best_policy_rows.append(
                    {
                        "dataset": regime_label,
                        "regime_index": int(regime_index),
                        "n_ports": int(n_ports),
                        "best_policy": str(best_row["policy"]),
                        "best_mean_observed_op": float(best_row["mean_observed_op"]),
                    }
                )
            continue

        policy_b_reps = list_completed_repetitions(POLICY_B_RUN_DIR / regime_label)
        policy_c_reps = list_completed_repetitions(POLICY_C_RUN_DIR / regime_label)
        common_reps = sorted(policy_b_reps.intersection(policy_c_reps))
        if not ALLOW_PARTIAL_POLICY_RUNS and len(common_reps) != shared_config["num_mc_reps"]:
            log_status(
                script_name,
                "skip",
                (
                    f"Skipping {regime_label}: found {len(common_reps)} common repetitions, "
                    f"expected {shared_config['num_mc_reps']}"
                ),
            )
            skipped_regimes.append(regime_label)
            continue
        if len(common_reps) == 0:
            log_status(script_name, "skip", f"Skipping {regime_label}: no common Policy B/C repetitions found")
            skipped_regimes.append(regime_label)
            continue

        log_status(script_name, "regime", f"Analyzing regime {regime_index + 1}/14: {regime_label}")
        regime_data = load_kms_regime(regime_label=regime_label, common_path=shared_config["common_dataset_path"])
        policy_b_rows = aggregate_result_rows_from_run(POLICY_B_RUN_DIR, regime_label)
        policy_c_rows = aggregate_result_rows_from_run(POLICY_C_RUN_DIR, regime_label)
        policy_b_index = index_policy_rows_by_rep_and_budget(policy_b_rows)
        policy_c_index = index_policy_rows_by_rep_and_budget(policy_c_rows)

        comparison_rows: list[dict[str, Any]] = []
        for rep_index in common_reps:
            rng = build_rep_rng(
                base_seed=shared_config["data_seed"],
                regime_index=regime_index,
                rep_index=rep_index,
            )
            _, eval_indices = split_design_eval_indices(
                n_samples=regime_data.shape[0],
                eval_fraction=shared_config["mc_eval_fraction"],
                rng=rng,
            )
            eval_binary = threshold_to_binary(
                regime_data[eval_indices],
                threshold=shared_config["threshold"],
                snr_linear=shared_config["snr_linear"],
            )
            ideal_op, n_ideal_outage = evaluate_ideal_ports(eval_binary)

            # Policy A is reconstructed directly from the same evaluation split
            # so every comparison uses perfectly matched Monte Carlo rows.
            policy_a_rows_by_budget: dict[int, dict[str, Any]] = {}
            for n_ports in shared_config["target_ports"]:
                policy_a_row = build_policy_a_row(
                    regime_label=regime_label,
                    regime_index=regime_index,
                    rep_index=rep_index,
                    n_ports=n_ports,
                    binary_eval=eval_binary,
                    total_ports=shared_config["total_ports"],
                    ideal_op=ideal_op,
                    n_ideal_outage=n_ideal_outage,
                )
                comparison_rows.append(policy_a_row)
                policy_a_rows_by_budget[int(n_ports)] = policy_a_row

            for policy_index in (policy_b_index, policy_c_index):
                for n_ports in shared_config["target_ports"]:
                    source_row = policy_index[(rep_index, int(n_ports))]
                    baseline_row = policy_a_rows_by_budget[int(n_ports)]
                    comparison_rows.append(
                        normalize_policy_row(
                            source_row,
                            gain_vs_policy_a=float(baseline_row["observed_op"]) - float(source_row["observed_op"]),
                        )
                    )

        summary_rows = summarize_comparison_rows(
            comparison_rows=comparison_rows,
            regime_label=regime_label,
            regime_index=regime_index,
        )
        write_csv_rows_atomic(combined_csv_path, COMPARISON_FIELDNAMES, comparison_rows)
        write_csv_rows_atomic(summary_csv_path, SUMMARY_FIELDNAMES, summary_rows)
        plot_regime_op_vs_budget(regime_output_dir=regime_output_dir, regime_label=regime_label, summary_rows=summary_rows)
        plot_regime_gain_vs_policy_a(
            regime_output_dir=regime_output_dir,
            regime_label=regime_label,
            summary_rows=summary_rows,
        )
        write_regime_summary_text(regime_output_dir=regime_output_dir, regime_label=regime_label, summary_rows=summary_rows)
        processed_regimes.append(regime_label)
        cross_regime_summary_rows.extend(summary_rows)

        for n_ports in shared_config["target_ports"]:
            candidate_rows = [row for row in summary_rows if int(row["n_ports"]) == int(n_ports)]
            best_row = min(candidate_rows, key=lambda item: float(item["mean_observed_op"]))
            best_policy_rows.append(
                {
                    "dataset": regime_label,
                    "regime_index": int(regime_index),
                    "n_ports": int(n_ports),
                    "best_policy": str(best_row["policy"]),
                    "best_mean_observed_op": float(best_row["mean_observed_op"]),
                }
            )

        del regime_data

    write_csv_rows_atomic(analysis_output_dir / "cross_regime_summary.csv", SUMMARY_FIELDNAMES, cross_regime_summary_rows)
    write_csv_rows_atomic(analysis_output_dir / "best_policy_by_regime_and_budget.csv", BEST_POLICY_FIELDNAMES, best_policy_rows)
    plot_best_policy_heatmap(
        output_dir=analysis_output_dir,
        cross_regime_summary_rows=cross_regime_summary_rows,
        target_ports=shared_config["target_ports"],
    )
    write_cross_regime_summary_text(
        output_dir=analysis_output_dir,
        cross_regime_summary_rows=cross_regime_summary_rows,
        skipped_regimes=skipped_regimes,
        target_ports=shared_config["target_ports"],
    )
    write_json_atomic(
        analysis_output_dir / "analysis_status.json",
        {
            "processed_regimes": processed_regimes,
            "skipped_regimes": skipped_regimes,
        },
    )

    return {
        "output_dir": str(analysis_output_dir),
        "processed_regimes": processed_regimes,
        "skipped_regimes": skipped_regimes,
    }


if __name__ == "__main__":
    run_configured_policy_analysis()
