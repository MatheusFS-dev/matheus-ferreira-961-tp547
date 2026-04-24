"""Run per-regime Monte Carlo simulations for Policy C on the KMS dataset.

This script is the executable entrypoint for the Policy C experiment. Policy C
is the greedy forward-selection policy that builds the sensing pattern one port
at a time, always choosing the next port that reduces observed-only outage the
most on the design subset of a given regime. The script runs the experiment
separately for each of the 14 KMS regimes and writes one CSV per repetition so
an interrupted run can continue from the last finished file.

What this script does:
- Loads one regime at a time from the KMS dataset.
- Computes a canonical full-regime greedy order to document which ports the
  policy prefers when it sees all rows from that regime.
- Repeats a Monte Carlo procedure `NUM_MC_REPS` times for the same regime:
  - randomly split rows into design and evaluation subsets,
  - build the Policy B exceedance scores used as the greedy tie-break,
  - run greedy forward selection on the design subset,
  - evaluate the first `N` selected ports for every budget in `TARGET_PORTS`
    on the evaluation subset,
  - save the repetition output immediately.
- Moves to the next regime and repeats the full workflow.

Step-by-step structure of the file:
1. Import block:
   The script imports the shared helper module rather than reimplementing
   dataset loading, evaluation, or resume logic locally.
2. Simulation configuration section:
   This is the control panel for the experiment. It contains the dataset root,
   output root, threshold, SNR, budgets, number of repetitions, and evaluation
   fraction.
3. Run-config builder:
   `build_policy_c_run_config()` serializes the simulation settings into
   `run_config.json` so reruns can verify compatibility before reusing any
   existing files.
4. Canonical greedy score writer:
   `build_full_data_port_score_rows()` writes a table showing the full-regime
   greedy step at which each port entered the canonical order.
5. Main Monte Carlo runner:
   `run_configured_policy_c_simulation()` orchestrates the regime loop,
   canonical-output generation, repetition skipping on resume, and repetition
   CSV writing.

How Policy C differs from Policy B:
- Policy B ranks ports independently by exceedance probability.
- Policy C looks at the current selected set and asks which remaining port
  covers the largest number of still-uncovered successful rows.
- This means Policy C explicitly tries to avoid redundancy between ports and
  prefers complementary selections.

How restart/resume works:
- Each regime writes its results to a separate subdirectory under `OUTPUT_DIR`.
- Each repetition is saved to `rep_XXXXX.csv`.
- If a repetition file already exists, it is treated as completed and skipped
  on the next run.
- `run_config.json` is checked before resuming so results are never mixed
  across incompatible settings.

Output layout:
- `run_config.json`: persisted simulation configuration.
- `<regime>/canonical_patterns.csv`: canonical full-regime Policy C patterns
  for the requested budgets.
- `<regime>/full_data_port_scores.csv`: one row per port with exceedance score,
  mean value, and greedy step.
- `<regime>/rep_XXXXX.csv`: one repetition result file with one row per budget.

Example command:
```bash
python src/architectures/dataset_analysis/policy_c_kms_per_regime_mc.py
```

Example customization:
- Change `NUM_MC_REPS` to run a shorter debug experiment or a longer final
  study.
- Change `RUN_NAME` to store the outputs in a new run directory.
- Change `MC_EVAL_FRACTION` if you want a different design/evaluation balance.

This script does not compare against Policy A and does not create plots. It
only generates the Policy C simulation files needed by the later analysis
script.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = REPO_ROOT / "src"
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from policy_mc_kms_common import (
    CANONICAL_FIELDNAMES,
    DEFAULT_DATASET_ROOT_PATH,
    KMS_REGIME_LABELS,
    RESULT_FIELDNAMES,
    build_canonical_rows,
    build_policy_b_order,
    build_policy_c_order,
    build_rep_rng,
    build_result_rows,
    ensure_directory,
    ensure_matching_run_config,
    list_completed_repetitions,
    load_kms_regime,
    log_status,
    repetition_result_path,
    split_design_eval_indices,
    threshold_to_binary,
    write_csv_rows_atomic,
)


# -----------------------------------------------------------------------------
# Simulation configuration
# -----------------------------------------------------------------------------
COMMON_DATASET_PATH = DEFAULT_DATASET_ROOT_PATH
RUNS_ROOT = REPO_ROOT / "runs" / "dataset_analysis"
RUN_NAME = "policy_c_kms_per_regime_mc"
OUTPUT_DIR = RUNS_ROOT / RUN_NAME
DATA_SEED = 0
THRESHOLD = 0.8
SNR_LINEAR = 1.0
TOTAL_PORTS = 100
TARGET_PORTS = (3, 4, 5, 6, 7)
NUM_MC_REPS = 100
MC_EVAL_FRACTION = 0.20


def build_policy_c_run_config() -> dict[str, Any]:
    """Build the persisted configuration for the Policy C simulation run.

    Args:
        None: This function reads the module-level configuration constants.

    Returns:
        dict[str, Any]: JSON-serializable configuration dictionary stored in
            the run directory to validate resumable reruns.

    Raises:
        ValueError: If ``TARGET_PORTS`` is empty.
    """
    if len(TARGET_PORTS) == 0:
        raise ValueError("TARGET_PORTS must contain at least one observation budget.")

    return {
        "script_name": Path(__file__).stem,
        "policy_name": "policy_c",
        "common_dataset_path": str(COMMON_DATASET_PATH),
        "output_dir": str(OUTPUT_DIR),
        "data_seed": int(DATA_SEED),
        "threshold": float(THRESHOLD),
        "snr_linear": float(SNR_LINEAR),
        "total_ports": int(TOTAL_PORTS),
        "target_ports": [int(value) for value in TARGET_PORTS],
        "num_mc_reps": int(NUM_MC_REPS),
        "mc_eval_fraction": float(MC_EVAL_FRACTION),
        "regime_labels": list(KMS_REGIME_LABELS),
    }


def build_full_data_port_score_rows(
    regime_label: str,
    regime_index: int,
    full_order: np.ndarray,
    exceedance_scores: np.ndarray,
    port_means: np.ndarray,
) -> list[dict[str, Any]]:
    """Build one full-data score table for the greedy Policy C run.

    Args:
        regime_label (str): Human-readable regime label.
        regime_index (int): Zero-based regime index.
        full_order (np.ndarray): Full greedy order from best first step to the
            final fallback-filled position.
        exceedance_scores (np.ndarray): Policy B exceedance scores on the full
            regime, used to interpret the greedy tie-break behavior.
        port_means (np.ndarray): Full-regime port means used as the last
            deterministic tie-break.

    Returns:
        list[dict[str, Any]]: One row per port containing scores and the greedy
            step at which the port entered the canonical order.

    Raises:
        ValueError: If the arrays do not align in length.
    """
    if full_order.shape[0] != exceedance_scores.shape[0] or full_order.shape[0] != port_means.shape[0]:
        raise ValueError("full_order, exceedance_scores, and port_means must have the same length.")

    step_lookup = np.empty(full_order.shape[0], dtype=int)
    step_lookup[full_order] = np.arange(1, full_order.shape[0] + 1, dtype=int)

    rows: list[dict[str, Any]] = []
    for port_index in range(full_order.shape[0]):
        rows.append(
            {
                "dataset": regime_label,
                "regime_index": int(regime_index),
                "port": int(port_index + 1),
                "exceed_prob": f"{float(exceedance_scores[port_index]):.10f}",
                "mean_value": f"{float(port_means[port_index]):.10f}",
                "policy_c_step": int(step_lookup[port_index]),
            }
        )
    return rows


def run_configured_policy_c_simulation() -> dict[str, Any]:
    """Run the per-regime Monte Carlo experiment for Policy C.

    Args:
        None: The simulation is configured entirely by the module-level
            constants declared near the top of the file.

    Returns:
        dict[str, Any]: Lightweight run summary containing the output
            directory and configured repetition count.

    Raises:
        ValueError: If the loaded regime width differs from ``TOTAL_PORTS`` or
            if an existing run directory contains incompatible configuration.
        FileNotFoundError: If any expected KMS regime file is missing.
        OSError: If result artifacts cannot be written.
    """
    script_name = Path(__file__).stem
    output_dir = ensure_directory(OUTPUT_DIR)
    ensure_matching_run_config(output_dir, build_policy_c_run_config())

    for regime_index, regime_label in enumerate(KMS_REGIME_LABELS):
        regime_output_dir = ensure_directory(output_dir / regime_label)
        log_status(script_name, "regime", f"Loading regime {regime_index + 1}/14: {regime_label}")
        regime_data = load_kms_regime(regime_label=regime_label, common_path=COMMON_DATASET_PATH)
        if regime_data.shape[1] != TOTAL_PORTS:
            raise ValueError(
                f"Regime {regime_label} has {regime_data.shape[1]} ports, expected {TOTAL_PORTS}."
            )

        full_binary = threshold_to_binary(regime_data, threshold=THRESHOLD, snr_linear=SNR_LINEAR)
        full_means = np.mean(regime_data, axis=0, dtype=np.float64)
        _, full_scores = build_policy_b_order(design_binary=full_binary, design_means=full_means)
        full_order = build_policy_c_order(
            design_binary=full_binary,
            design_means=full_means,
            policy_b_scores=full_scores,
            max_budget=TOTAL_PORTS,
        )
        write_csv_rows_atomic(
            regime_output_dir / "canonical_patterns.csv",
            fieldnames=CANONICAL_FIELDNAMES,
            rows=build_canonical_rows(
                policy_name="policy_c",
                regime_label=regime_label,
                regime_index=regime_index,
                full_order=full_order,
                target_ports=TARGET_PORTS,
            ),
        )
        write_csv_rows_atomic(
            regime_output_dir / "full_data_port_scores.csv",
            fieldnames=("dataset", "regime_index", "port", "exceed_prob", "mean_value", "policy_c_step"),
            rows=build_full_data_port_score_rows(
                regime_label=regime_label,
                regime_index=regime_index,
                full_order=full_order,
                exceedance_scores=full_scores,
                port_means=full_means,
            ),
        )
        del full_binary

        completed_reps = list_completed_repetitions(regime_output_dir)
        for rep_index in range(NUM_MC_REPS):
            if rep_index in completed_reps:
                log_status(script_name, "resume", f"Skipping {regime_label} repetition {rep_index}: already done")
                continue

            rng = build_rep_rng(base_seed=DATA_SEED, regime_index=regime_index, rep_index=rep_index)
            design_indices, eval_indices = split_design_eval_indices(
                n_samples=regime_data.shape[0],
                eval_fraction=MC_EVAL_FRACTION,
                rng=rng,
            )

            # Policy C re-learns the greedy order in each repetition so the
            # saved Monte Carlo variability reflects sensitivity to dataset
            # resampling rather than only evaluation noise.
            design_values = regime_data[design_indices]
            eval_values = regime_data[eval_indices]
            design_binary = threshold_to_binary(design_values, threshold=THRESHOLD, snr_linear=SNR_LINEAR)
            eval_binary = threshold_to_binary(eval_values, threshold=THRESHOLD, snr_linear=SNR_LINEAR)
            design_means = np.mean(design_values, axis=0, dtype=np.float64)
            _, policy_b_scores = build_policy_b_order(design_binary=design_binary, design_means=design_means)
            policy_order = build_policy_c_order(
                design_binary=design_binary,
                design_means=design_means,
                policy_b_scores=policy_b_scores,
                max_budget=max(TARGET_PORTS),
            )

            rep_rows = build_result_rows(
                policy_name="policy_c",
                regime_label=regime_label,
                regime_index=regime_index,
                rep_index=rep_index,
                base_seed=DATA_SEED,
                design_size=int(design_indices.shape[0]),
                eval_size=int(eval_indices.shape[0]),
                full_order=policy_order,
                target_ports=TARGET_PORTS,
                binary_eval=eval_binary,
            )
            write_csv_rows_atomic(repetition_result_path(regime_output_dir, rep_index), RESULT_FIELDNAMES, rep_rows)
            log_status(script_name, "rep", f"Saved {regime_label} repetition {rep_index} results")
            del design_values, eval_values, design_binary, eval_binary, design_means, policy_b_scores, policy_order, rep_rows

        del regime_data

    return {
        "output_dir": str(output_dir),
        "num_mc_reps": int(NUM_MC_REPS),
        "target_ports": tuple(int(value) for value in TARGET_PORTS),
    }


if __name__ == "__main__":
    run_configured_policy_c_simulation()
