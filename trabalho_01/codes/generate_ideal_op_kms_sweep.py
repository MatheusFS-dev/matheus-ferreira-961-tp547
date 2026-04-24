"""Generate per-regime ideal outage probability sweeps on the KMS dataset.

This script is a standalone utility dedicated to the ideal outage probability
of each KMS regime. Unlike the Policy B and Policy C simulators, this file
does not perform Monte Carlo design/evaluation resampling or learn any sensing
pattern. Its purpose is to answer a simpler question:

"If all 100 ports were perfectly observable, what would the ideal outage
probability be for each regime under different thresholds and/or SNR values?"

What this script does:
- Load one KMS regime at a time.
- For every configured pair of `(threshold, snr_linear)` values:
  - convert the float-valued regime matrix into a binary exceedance matrix,
  - compute the ideal outage probability from the full 100-port observation,
  - compute the corresponding Wilson confidence interval,
  - save the results to a per-regime CSV file.
- Generate one plot per regime:
  - a line plot when only one of threshold or SNR is swept,
  - a heatmap when both threshold and SNR vary.
- Write one combined CSV with all regimes at the end.

Step-by-step structure of the file:
1. Import block:
   The script resolves the repository root, adds `src` to `sys.path`, and
   imports the shared helper module used by the other dataset-analysis files.
2. Configuration section:
   This is the main control panel for the sweep. You can change the dataset
   path, output directory, thresholds, SNR values, and whether existing files
   should be rebuilt.
3. Run-config builder:
   `build_ideal_op_run_config()` serializes the current settings into
   `run_config.json` so reruns can resume safely only when the configuration
   matches the files already on disk.
4. Row builder:
   `build_regime_ideal_rows()` computes one output row per `(threshold, snr)`
   pair for one regime.
5. Plot helpers:
   The plotting functions adapt automatically to the requested sweep shape.
6. Main runner:
   `run_configured_ideal_op_sweep()` loops over the 14 regimes, skips already
   completed regime CSVs on resume, writes per-regime artifacts, and finally
   writes one combined CSV across all regimes.

How restart/resume works:
- Each regime has its own subdirectory under `OUTPUT_DIR`.
- If `ideal_op_sweep.csv` already exists for a regime and
  `FORCE_REBUILD = False`, that regime is skipped and its existing rows are
  reused when rebuilding the combined CSV.
- The script also validates `run_config.json` before reusing any files, which
  prevents mixing outputs generated with different thresholds, SNR grids, or
  dataset paths.

Output layout:
- `run_config.json`: persisted configuration for the sweep.
- `<regime>/ideal_op_sweep.csv`: one row per `(threshold, snr_linear)` pair.
- `<regime>/ideal_op_plot.png`: line plot or heatmap depending on the sweep.
- `<regime>/summary.txt`: compact text summary for the regime.
- `ideal_op_all_regimes.csv`: combined results across all regimes.

Example commands:
```bash
python src/architectures/dataset_analysis/generate_ideal_op_kms_sweep.py
```

Example configuration ideas:
- Sweep only thresholds:
  - `THRESHOLD_VALUES = (0.2, 0.4, 0.6, 0.8, 1.0)`
  - `SNR_LINEAR_VALUES = (1.0,)`
- Sweep only SNR:
  - `THRESHOLD_VALUES = (0.8,)`
  - `SNR_LINEAR_VALUES = (0.5, 1.0, 2.0, 4.0)`
- Sweep both:
  - `THRESHOLD_VALUES = (0.4, 0.8, 1.2)`
  - `SNR_LINEAR_VALUES = (0.5, 1.0, 2.0)`

This script is intentionally independent from the Policy B / Policy C
simulation workflow. It focuses only on regime-level ideal OP baselines.
"""

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = REPO_ROOT / "src"
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from policy_mc_kms_common import (
    compute_wilson_interval,
    DEFAULT_DATASET_ROOT_PATH,
    ensure_directory,
    ensure_matching_run_config,
    evaluate_ideal_ports,
    KMS_REGIME_LABELS,
    load_kms_regime,
    log_status,
    read_csv_rows,
    threshold_to_binary,
    write_csv_rows_atomic,
)


# -----------------------------------------------------------------------------
# Sweep configuration
# -----------------------------------------------------------------------------
COMMON_DATASET_PATH = DEFAULT_DATASET_ROOT_PATH
RUNS_ROOT = REPO_ROOT / "runs" / "dataset_analysis"
RUN_NAME = "ideal_op_kms_sweep"
OUTPUT_DIR = RUNS_ROOT / RUN_NAME
FORCE_REBUILD = False
THRESHOLD_VALUES = (0.2, 0.4, 0.6, 0.8, 1.0)
SNR_LINEAR_VALUES = (1.0,)


IDEAL_OP_FIELDNAMES: tuple[str, ...] = (
    "dataset",
    "regime_index",
    "threshold",
    "snr_linear",
    "threshold_over_snr",
    "n_samples",
    "n_ideal_outage",
    "ideal_op",
    "wilson_low",
    "wilson_high",
)


def build_ideal_op_run_config() -> dict[str, Any]:
    """Build the persisted run configuration for the ideal-OP sweep.

    Args:
        None: This function uses only the module-level configuration values.

    Returns:
        dict[str, Any]: JSON-serializable configuration dictionary describing
            the current sweep.

    Raises:
        ValueError: If either configured value sequence is empty.
    """
    if len(THRESHOLD_VALUES) == 0:
        raise ValueError("THRESHOLD_VALUES must contain at least one value.")
    if len(SNR_LINEAR_VALUES) == 0:
        raise ValueError("SNR_LINEAR_VALUES must contain at least one value.")

    return {
        "script_name": Path(__file__).stem,
        "common_dataset_path": str(COMMON_DATASET_PATH),
        "output_dir": str(OUTPUT_DIR),
        "force_rebuild": bool(FORCE_REBUILD),
        "threshold_values": [float(value) for value in THRESHOLD_VALUES],
        "snr_linear_values": [float(value) for value in SNR_LINEAR_VALUES],
        "regime_labels": list(KMS_REGIME_LABELS),
    }


def build_regime_ideal_rows(
    regime_label: str,
    regime_index: int,
    regime_data: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute the ideal-OP sweep rows for one regime.

    Args:
        regime_label (str): Human-readable regime label.
        regime_index (int): Zero-based regime index in the canonical KMS order.
        regime_data (np.ndarray): Float-valued regime matrix with shape
            ``(n_samples, 100)``.

    Returns:
        list[dict[str, Any]]: One row per configured `(threshold, snr_linear)`
            pair, aligned with ``IDEAL_OP_FIELDNAMES``.

    Raises:
        ValueError: If ``regime_data`` is not a non-empty 2D matrix.
    """
    if regime_data.ndim != 2 or regime_data.shape[0] == 0:
        raise ValueError("regime_data must be a non-empty 2D matrix.")

    rows: list[dict[str, Any]] = []
    n_samples = int(regime_data.shape[0])

    # The ideal OP depends only on whether any of the 100 ports exceeds the
    # threshold-adjusted cutoff. For each threshold/SNR pair we therefore
    # threshold the full regime matrix and evaluate the ideal full-port success.
    for threshold_value in THRESHOLD_VALUES:
        for snr_value in SNR_LINEAR_VALUES:
            binary_regime = threshold_to_binary(
                regime_data,
                threshold=float(threshold_value),
                snr_linear=float(snr_value),
            )
            ideal_op, n_ideal_outage = evaluate_ideal_ports(binary_regime)
            wilson_low, wilson_high = compute_wilson_interval(
                n_outage=n_ideal_outage,
                n_total=n_samples,
            )
            rows.append(
                {
                    "dataset": regime_label,
                    "regime_index": int(regime_index),
                    "threshold": float(threshold_value),
                    "snr_linear": float(snr_value),
                    "threshold_over_snr": float(threshold_value / snr_value),
                    "n_samples": n_samples,
                    "n_ideal_outage": int(n_ideal_outage),
                    "ideal_op": float(ideal_op),
                    "wilson_low": float(wilson_low),
                    "wilson_high": float(wilson_high),
                }
            )
    return rows


def plot_regime_ideal_sweep(regime_output_dir: Path, regime_label: str, rows: list[dict[str, Any]]) -> Path:
    """Plot the ideal-OP sweep for one regime.

    Args:
        regime_output_dir (Path): Output directory for the regime.
        regime_label (str): Human-readable regime label.
        rows (list[dict[str, Any]]): Sweep rows produced by
            ``build_regime_ideal_rows``.

    Returns:
        Path: Saved PNG path.

    Raises:
        OSError: If the figure cannot be written.
    """
    thresholds = np.array(sorted(set(float(row["threshold"]) for row in rows)), dtype=np.float64)
    snr_values = np.array(sorted(set(float(row["snr_linear"]) for row in rows)), dtype=np.float64)
    output_path = regime_output_dir / "ideal_op_plot.png"

    if len(snr_values) == 1:
        x_values = thresholds
        y_values = np.array(
            [
                float(
                    next(
                        row["ideal_op"]
                        for row in rows
                        if float(row["threshold"]) == float(threshold_value)
                        and float(row["snr_linear"]) == float(snr_values[0])
                    )
                )
                for threshold_value in thresholds
            ],
            dtype=np.float64,
        )
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        ax.plot(x_values, y_values, color="tab:blue", marker="o", linewidth=2.0)
        ax.set_title(f"Ideal OP vs threshold: {regime_label}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Ideal outage probability")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    if len(thresholds) == 1:
        x_values = snr_values
        y_values = np.array(
            [
                float(
                    next(
                        row["ideal_op"]
                        for row in rows
                        if float(row["threshold"]) == float(thresholds[0])
                        and float(row["snr_linear"]) == float(snr_value)
                    )
                )
                for snr_value in snr_values
            ],
            dtype=np.float64,
        )
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        ax.plot(x_values, y_values, color="tab:orange", marker="s", linewidth=2.0)
        ax.set_title(f"Ideal OP vs SNR: {regime_label}")
        ax.set_xlabel("SNR (linear)")
        ax.set_ylabel("Ideal outage probability")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path

    # When both dimensions vary, a heatmap is more compact and makes the
    # threshold/SNR interaction visible at a glance.
    heatmap = np.empty((len(thresholds), len(snr_values)), dtype=np.float64)
    for threshold_index, threshold_value in enumerate(thresholds):
        for snr_index, snr_value in enumerate(snr_values):
            heatmap[threshold_index, snr_index] = float(
                next(
                    row["ideal_op"]
                    for row in rows
                    if float(row["threshold"]) == float(threshold_value)
                    and float(row["snr_linear"]) == float(snr_value)
                )
            )

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    image = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_title(f"Ideal OP heatmap: {regime_label}")
    ax.set_xlabel("SNR (linear)")
    ax.set_ylabel("Threshold")
    ax.set_xticks(np.arange(len(snr_values)))
    ax.set_xticklabels([f"{value:.3g}" for value in snr_values])
    ax.set_yticks(np.arange(len(thresholds)))
    ax.set_yticklabels([f"{value:.3g}" for value in thresholds])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Ideal outage probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def write_regime_summary_text(regime_output_dir: Path, regime_label: str, rows: list[dict[str, Any]]) -> Path:
    """Write a concise text summary of the ideal-OP sweep for one regime.

    Args:
        regime_output_dir (Path): Output directory for the regime.
        regime_label (str): Human-readable regime label.
        rows (list[dict[str, Any]]): Sweep rows for the regime.

    Returns:
        Path: Saved summary text file path.

    Raises:
        OSError: If the file cannot be written.
    """
    best_row = min(rows, key=lambda item: float(item["ideal_op"]))
    worst_row = max(rows, key=lambda item: float(item["ideal_op"]))
    lines = [
        f"Regime: {regime_label}",
        "",
        (
            "Best ideal OP: "
            f"{float(best_row['ideal_op']):.8f} at threshold={float(best_row['threshold']):.6g}, "
            f"snr_linear={float(best_row['snr_linear']):.6g}"
        ),
        (
            "Worst ideal OP: "
            f"{float(worst_row['ideal_op']):.8f} at threshold={float(worst_row['threshold']):.6g}, "
            f"snr_linear={float(worst_row['snr_linear']):.6g}"
        ),
        f"Evaluated combinations: {len(rows)}",
    ]
    output_path = regime_output_dir / "summary.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_configured_ideal_op_sweep() -> dict[str, Any]:
    """Run the per-regime ideal-OP sweep across thresholds and/or SNR values.

    Args:
        None: The sweep is controlled entirely by the module-level constants.

    Returns:
        dict[str, Any]: Summary containing the output directory and processed
            regime labels.

    Raises:
        FileNotFoundError: If any configured regime file is missing.
        ValueError: If the run configuration does not match an existing output
            directory or if a regime matrix has invalid shape.
        OSError: If result files cannot be written.
    """
    script_name = Path(__file__).stem
    output_dir = ensure_directory(OUTPUT_DIR)
    ensure_matching_run_config(output_dir, build_ideal_op_run_config())

    all_rows: list[dict[str, Any]] = []
    processed_regimes: list[str] = []

    for regime_index, regime_label in enumerate(KMS_REGIME_LABELS):
        regime_output_dir = ensure_directory(output_dir / regime_label)
        regime_csv_path = regime_output_dir / "ideal_op_sweep.csv"

        if regime_csv_path.exists() and not FORCE_REBUILD:
            log_status(script_name, "resume", f"Skipping {regime_label}: ideal OP sweep already exists")
            existing_rows = []
            for row in read_csv_rows(regime_csv_path):
                existing_rows.append(
                    {
                        "dataset": str(row["dataset"]),
                        "regime_index": int(row["regime_index"]),
                        "threshold": float(row["threshold"]),
                        "snr_linear": float(row["snr_linear"]),
                        "threshold_over_snr": float(row["threshold_over_snr"]),
                        "n_samples": int(row["n_samples"]),
                        "n_ideal_outage": int(row["n_ideal_outage"]),
                        "ideal_op": float(row["ideal_op"]),
                        "wilson_low": float(row["wilson_low"]),
                        "wilson_high": float(row["wilson_high"]),
                    }
                )
            all_rows.extend(existing_rows)
            processed_regimes.append(regime_label)
            continue

        log_status(script_name, "regime", f"Processing regime {regime_index + 1}/14: {regime_label}")
        regime_data = load_kms_regime(regime_label=regime_label, common_path=COMMON_DATASET_PATH)
        regime_rows = build_regime_ideal_rows(
            regime_label=regime_label,
            regime_index=regime_index,
            regime_data=regime_data,
        )
        write_csv_rows_atomic(regime_csv_path, IDEAL_OP_FIELDNAMES, regime_rows)
        plot_regime_ideal_sweep(regime_output_dir=regime_output_dir, regime_label=regime_label, rows=regime_rows)
        write_regime_summary_text(regime_output_dir=regime_output_dir, regime_label=regime_label, rows=regime_rows)

        all_rows.extend(regime_rows)
        processed_regimes.append(regime_label)
        del regime_data

    write_csv_rows_atomic(output_dir / "ideal_op_all_regimes.csv", IDEAL_OP_FIELDNAMES, all_rows)
    return {
        "output_dir": str(output_dir),
        "processed_regimes": processed_regimes,
        "num_thresholds": int(len(THRESHOLD_VALUES)),
        "num_snr_values": int(len(SNR_LINEAR_VALUES)),
    }


if __name__ == "__main__":
    run_configured_ideal_op_sweep()
