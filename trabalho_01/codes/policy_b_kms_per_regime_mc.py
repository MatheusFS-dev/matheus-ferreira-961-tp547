"""Run per-regime Monte Carlo simulations for Policy B on the KMS dataset.

This script is the executable entrypoint for the Policy B experiment. Policy B
is the threshold-aware ranking policy that scores each port by how often it
exceeds the outage threshold inside one regime. The script runs the experiment
independently for each of the 14 KMS regimes and saves one CSV file per Monte
Carlo repetition so the run can resume after interruptions.

What this script does:
- Loads one KMS regime at a time.
- Builds a full-regime canonical Policy B ranking for documentation.
- Repeats a Monte Carlo procedure `NUM_MC_REPS` times for that regime:
  - randomly split rows into design and evaluation subsets,
  - estimate port exceedance probabilities on the design subset,
  - rank ports from best to worst,
  - evaluate the first `N` ranked ports for every budget in `TARGET_PORTS`
    on the evaluation subset,
  - write the repetition results to disk immediately.
- Moves to the next regime and repeats the same process.

Step-by-step structure of the file:
1. Import block:
   The script resolves `src` dynamically and imports the shared helper module.
   This keeps the executable small and ensures it uses the exact same Monte
   Carlo and file-format logic as the other scripts.
2. Simulation configuration section:
   This is the part you are expected to edit when changing the dataset path,
   output root, budgets, threshold, number of Monte Carlo repetitions, or
   evaluation fraction. All high-level experiment knobs live together here.
3. Run-config builder:
   `build_policy_b_run_config()` converts the Python constants into a JSON
   document stored inside the run directory. On restart, the script checks that
   the existing configuration matches the current one before continuing.
4. Canonical ranking writer:
   `build_full_data_port_score_rows()` creates a regime-level explanation table
   showing the full-data Policy B ranking, exceedance probability, and mean
   value of every port.
5. Main Monte Carlo runner:
   `run_configured_policy_b_simulation()` is the orchestration function. It
   loops over the 14 regimes, writes canonical outputs, skips completed
   repetitions when resuming, and saves each new repetition as a standalone
   CSV file.

How restart/resume works:
- Each regime has its own output folder under `OUTPUT_DIR`.
- Each repetition is written to a file named like `rep_00007.csv`.
- On a rerun, the script scans the regime folder and skips any repetition whose
  CSV already exists.
- The script also validates `run_config.json` before resuming so results from
  different simulation settings are never mixed accidentally.

Output layout:
- `run_config.json`: persisted simulation configuration.
- `<regime>/canonical_patterns.csv`: canonical full-regime Policy B patterns
  for each requested budget.
- `<regime>/full_data_port_scores.csv`: full ranking table for all 100 ports.
- `<regime>/rep_XXXXX.csv`: one repetition result file containing one row per
  observation budget.

Example command:
```bash
python src/architectures/dataset_analysis/policy_b_kms_per_regime_mc.py
```

Example customization:
- Change `RUN_NAME` to create a separate run folder.
- Change `COMMON_DATASET_PATH` if the dataset is mounted in a different
  location.
- Change `TARGET_PORTS` to evaluate a different set of observation budgets.

The script does not generate plots. It only runs the Policy B simulations and
stores the raw result files that will later be consumed by the analysis script.
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
RUN_NAME = "policy_b_kms_per_regime_mc"
OUTPUT_DIR = RUNS_ROOT / RUN_NAME
DATA_SEED = 0
THRESHOLD = 0.8
SNR_LINEAR = 1.0
TOTAL_PORTS = 100
TARGET_PORTS = (3, 4, 5, 6, 7)
NUM_MC_REPS = 100
MC_EVAL_FRACTION = 0.20


def build_policy_b_run_config() -> dict[str, Any]:
    """Build the persisted configuration for the Policy B simulation run.

    Args:
        None: This function reads the module-level configuration constants.

    Returns:
        dict[str, Any]: JSON-serializable configuration dictionary used to
            validate restarts and document the run directory contents.

    Raises:
        ValueError: If ``TARGET_PORTS`` is empty.
    """
    if len(TARGET_PORTS) == 0:
        raise ValueError("TARGET_PORTS must contain at least one observation budget.")

    return {
        "script_name": Path(__file__).stem,
        "policy_name": "policy_b",
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
    """Build one full-data Policy B ranking table for a regime.

    Args:
        regime_label (str): Human-readable regime label.
        regime_index (int): Zero-based regime index.
        full_order (np.ndarray): Complete Policy B ranking from best to worst.
        exceedance_scores (np.ndarray): Per-port exceedance frequencies used as
            the primary ranking score.
        port_means (np.ndarray): Per-port means used as the deterministic
            secondary tie-break.

    Returns:
        list[dict[str, Any]]: One row per port containing the ranking features
            and the final full-data rank.

    Raises:
        ValueError: If the three arrays do not align in length.
    """
    if full_order.shape[0] != exceedance_scores.shape[0] or full_order.shape[0] != port_means.shape[0]:
        raise ValueError("full_order, exceedance_scores, and port_means must have the same length.")

    rank_lookup = np.empty(full_order.shape[0], dtype=int)
    rank_lookup[full_order] = np.arange(1, full_order.shape[0] + 1, dtype=int)

    # Persisting the full-data ranking helps interpret why a regime preferred
    # specific ports, even though the actual Monte Carlo simulation re-learns
    # the ranking inside each repetition.
    rows: list[dict[str, Any]] = []
    for port_index in range(full_order.shape[0]):
        rows.append(
            {
                "dataset": regime_label,
                "regime_index": int(regime_index),
                "port": int(port_index + 1),
                "exceed_prob": f"{float(exceedance_scores[port_index]):.10f}",
                "mean_value": f"{float(port_means[port_index]):.10f}",
                "policy_b_rank": int(rank_lookup[port_index]),
            }
        )
    return rows


def run_configured_policy_b_simulation() -> dict[str, Any]:
    """Run the per-regime Monte Carlo experiment for Policy B.

    Args:
        None: The function is fully controlled by the module-level simulation
            constants defined near the top of the file.

    Returns:
        dict[str, Any]: Lightweight run summary containing the output
            directory and the number of configured repetitions.

    Raises:
        ValueError: If the loaded regime width differs from ``TOTAL_PORTS`` or
            if an existing run directory contains incompatible configuration.
        FileNotFoundError: If any expected KMS regime file is missing.
        OSError: If result artifacts cannot be written.
    """
    script_name = Path(__file__).stem
    output_dir = ensure_directory(OUTPUT_DIR)
    ensure_matching_run_config(output_dir, build_policy_b_run_config())

    for regime_index, regime_label in enumerate(KMS_REGIME_LABELS):
        regime_output_dir = ensure_directory(output_dir / regime_label)
        log_status(script_name, "regime", f"Loading regime {regime_index + 1}/14: {regime_label}")
        regime_data = load_kms_regime(regime_label=regime_label, common_path=COMMON_DATASET_PATH)
        if regime_data.shape[1] != TOTAL_PORTS:
            raise ValueError(
                f"Regime {regime_label} has {regime_data.shape[1]} ports, expected {TOTAL_PORTS}."
            )

        # Save one canonical full-regime ranking so the run directory remains
        # interpretable even without opening every repetition artifact.
        full_binary = threshold_to_binary(regime_data, threshold=THRESHOLD, snr_linear=SNR_LINEAR)
        full_means = np.mean(regime_data, axis=0, dtype=np.float64)
        full_order, full_scores = build_policy_b_order(design_binary=full_binary, design_means=full_means)
        write_csv_rows_atomic(
            regime_output_dir / "canonical_patterns.csv",
            fieldnames=CANONICAL_FIELDNAMES,
            rows=build_canonical_rows(
                policy_name="policy_b",
                regime_label=regime_label,
                regime_index=regime_index,
                full_order=full_order,
                target_ports=TARGET_PORTS,
            ),
        )
        write_csv_rows_atomic(
            regime_output_dir / "full_data_port_scores.csv",
            fieldnames=("dataset", "regime_index", "port", "exceed_prob", "mean_value", "policy_b_rank"),
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

            # The design split learns the ranking for this repetition; the
            # evaluation split remains unseen so the outage estimate stays
            # faithful to the Monte Carlo protocol.
            design_values = regime_data[design_indices]
            eval_values = regime_data[eval_indices]
            design_binary = threshold_to_binary(design_values, threshold=THRESHOLD, snr_linear=SNR_LINEAR)
            eval_binary = threshold_to_binary(eval_values, threshold=THRESHOLD, snr_linear=SNR_LINEAR)
            design_means = np.mean(design_values, axis=0, dtype=np.float64)
            policy_order, _ = build_policy_b_order(design_binary=design_binary, design_means=design_means)

            rep_rows = build_result_rows(
                policy_name="policy_b",
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
            del design_values, eval_values, design_binary, eval_binary, design_means, policy_order, rep_rows

        # Explicitly dropping the loaded regime helps long runs keep memory
        # bounded to one regime at a time.
        del regime_data

    return {
        "output_dir": str(output_dir),
        "num_mc_reps": int(NUM_MC_REPS),
        "target_ports": tuple(int(value) for value in TARGET_PORTS),
    }


if __name__ == "__main__":
    run_configured_policy_b_simulation()
