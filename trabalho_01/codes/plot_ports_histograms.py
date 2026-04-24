"""Generate per-regime and global port-statistics plots for the dataset.

This script performs a focused analysis for answering the question:
"Where are the greatest values by index?"

Execution flow:
1. Load datasets with either ``load_data`` or ``load_data_all`` (global switch).
2. Validate shape consistency across all loaded subsets.
3. For each individual dataset/regime, compute per-port statistics and save:
   ``<output>/per_regime/<regime>/port_statistics.csv``
   ``<output>/per_regime/<regime>/mean_x_ports.png``
   ``<output>/per_regime/<regime>/max_x_ports.png``
   ``<output>/per_regime/<regime>/std_x_ports.png``
   ``<output>/per_regime/<regime>/max_histogram.png``
4. Compute one additional global analysis using all loaded datasets together:
   ``<output>/global/port_statistics.csv``
   ``<output>/global/mean_x_ports.png``
   ``<output>/global/max_x_ports.png``
   ``<output>/global/std_x_ports.png``
   ``<output>/global/max_histogram.png``
5. Save one manifest CSV at the output root summarizing every generated group.

Interpretation of saved figures:
- MeanXports: high peaks indicate high average ports.
- MaxXports: high peaks indicate extreme best-case ports.
- StdXports: high peaks indicate unstable ports with larger spread.
- Max histogram: shows max value for each port index.

This file intentionally does not generate combined overlays with Mean ± Std or
ranking bar charts. The focus is on keeping one clean folder per regime and one
clean folder for the global aggregated view.
"""

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = REPO_ROOT / "src"
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))


from utils.data import load_raw_rr, load_raw_kms

# Global switch requested by the user:
# - True: use ``load_data_all`` (all 14 datasets).
# - False: use ``load_data`` (2 baseline datasets).
USE_LOAD_DATA_ALL = True

# Optional dataset root override. Keep ``None`` to rely on each loader default.
COMMON_DATA_PATH = None

# Output paths.
OUTPUT_DIR = Path("./runs/ports")


def _load_selected_datasets(use_load_data_all: bool, common_path: str | None) -> tuple[list[str], list[np.ndarray]]:
    """Load datasets according to the configured loader mode.

    Args:
        use_load_data_all (bool): Controls which helper is used.
            If ``True``, this function calls ``load_data_all`` and returns all
            14 kappa-mu shadowed subsets. This provides broader coverage but
            usually requires more memory and runtime.
            If ``False``, this function calls ``load_data`` and returns only the
            two baseline subsets (Rayleigh and Rician), which is lighter and
            faster but less comprehensive.
        common_path (str | None): Base path passed to the selected loader.
            If ``None``, each loader uses its internal default path. If a custom
            string is provided, the loader resolves dataset files relative to it.

    Returns:
        tuple[list[str], list[np.ndarray]]: A pair containing dataset names and
            the corresponding arrays.

    Raises:
        ValueError: If no datasets were returned by the selected loader.

    Examples:
        >>> names, arrays = _load_selected_datasets(True, None)
        >>> len(names) == len(arrays)
        True
    """
    loader_name = "load_data_all" if use_load_data_all else "load_data"
    print(f"[1/6] Loading datasets with {loader_name}...")

    if use_load_data_all:
        dataset_names = [
            "kappa0_mu1_m0",
            "kappa0_mu1_m2",
            "kappa0_mu1_m50",
            "kappa0_mu2_m50",
            "kappa0_mu5_m50",
            "kappa5_mu1_m0",
            "kappa5_mu1_m2",
            "kappa5_mu1_m50",
            "kappa5_mu2_m0",
            "kappa5_mu2_m2",
            "kappa5_mu2_m50",
            "kappa5_mu5_m0",
            "kappa5_mu5_m2",
            "kappa5_mu5_m50",
        ]
        dataset_values = list(load_raw_kms(common_path=common_path))
    else:
        dataset_names = ["kappa0_mu1_m50", "kappa5_mu1_m50"]
        dataset_values = list(load_raw_rr(common_path=common_path))

    if not dataset_values:
        raise ValueError("Selected loader returned no dataset arrays.")

    print(f"Loaded {len(dataset_values)} datasets:")
    for name, data in zip(dataset_names, dataset_values, strict=True):
        print(f"  - {name}: shape={data.shape}")

    return dataset_names, dataset_values


def _sanitize_group_name(group_name: str) -> str:
    """Convert one dataset label into a filesystem-safe folder name.

    Args:
        group_name (str): Human-readable dataset or regime label. Characters
            that are awkward in filesystem paths are replaced with underscores.

    Returns:
        str: Sanitized folder-safe name. If the input contains only filtered
            characters, the fallback name ``"group"`` is returned.

    Raises:
        ValueError: If ``group_name`` is empty after trimming whitespace.

    Examples:
        >>> _sanitize_group_name("kappa0_mu1_m50")
        'kappa0_mu1_m50'
        >>> _sanitize_group_name("global view")
        'global_view'
    """
    trimmed_name = group_name.strip()
    if not trimmed_name:
        raise ValueError("group_name must not be empty.")

    safe_chars = []
    for character in trimmed_name:
        if character.isalnum() or character in {"_", "-"}:
            safe_chars.append(character)
        else:
            safe_chars.append("_")

    sanitized_name = "".join(safe_chars).strip("_")
    return sanitized_name or "group"


def _validate_datasets_and_get_global_range(
    dataset_names: list[str], dataset_values: list[np.ndarray]
) -> tuple[int, int, float, float]:
    """Validate loaded datasets and return global shape/range metadata.

    Args:
        dataset_names (list[str]): Human-readable names aligned with
            ``dataset_values``. These names are used in validation errors.
        dataset_values (list[np.ndarray]): Dataset matrices expected to be
            2D arrays with shape ``(n_samples, n_ports)``.

    Returns:
        tuple[int, int, float, float]: A tuple with
            ``(n_ports, total_samples, global_min, global_max)``.

    Raises:
        ValueError: If any dataset is not 2D or if the number of ports differs
            across datasets.

    Examples:
        >>> _validate_datasets_and_get_global_range(["a"], [np.ones((2, 3))])[:2]
        (3, 2)
    """
    print("[2/5] Validating datasets and collecting global metadata...")

    if len(dataset_names) != len(dataset_values):
        raise ValueError("Dataset names and values must have matching lengths.")

    n_ports_reference: int | None = None
    total_samples = 0
    global_min = np.inf
    global_max = -np.inf

    # Validate each subset once and extract metadata needed by downstream steps.
    for name, data in zip(dataset_names, dataset_values, strict=True):
        if data.ndim != 2:
            raise ValueError(f"Dataset '{name}' must be 2D, got shape {data.shape}.")

        n_ports_current = data.shape[1]
        if n_ports_reference is None:
            n_ports_reference = n_ports_current
        elif n_ports_current != n_ports_reference:
            raise ValueError(
                "All datasets must have the same number of ports. "
                f"Expected {n_ports_reference}, got {n_ports_current} in '{name}'."
            )

        total_samples += data.shape[0]
        subset_min = float(np.min(data))
        subset_max = float(np.max(data))
        global_min = min(global_min, subset_min)
        global_max = max(global_max, subset_max)

    if n_ports_reference is None:
        raise ValueError("No datasets available after validation.")

    print(
        f"Validated {len(dataset_values)} datasets with {total_samples} samples, "
        f"{n_ports_reference} ports, value range [{global_min}, {global_max}]"
    )
    return n_ports_reference, total_samples, global_min, global_max


def _build_port_statistics(
    dataset_values: list[np.ndarray],
    n_ports: int,
    total_samples: int,
) -> pd.DataFrame:
    """Compute memory-efficient per-port descriptive statistics.

    Args:
        dataset_values (list[np.ndarray]): Loaded datasets with matching number
            of ports and potentially different number of samples.
        n_ports (int): Number of ports/features shared by all datasets.
        total_samples (int): Total number of rows across all datasets.

    Returns:
        pd.DataFrame: Per-port statistics with one row per port.

    Raises:
        ValueError: If ``total_samples`` is not positive.

    Examples:
        >>> d1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> _build_port_statistics([d1], n_ports=2, total_samples=2).shape[0]
        2
    """
    print("[3/5] Computing per-port descriptive statistics...")

    if total_samples <= 0:
        raise ValueError("total_samples must be greater than zero.")

    sum_values = np.zeros(n_ports, dtype=np.float64)
    sum_squares = np.zeros(n_ports, dtype=np.float64)
    min_values = np.full(n_ports, np.inf, dtype=np.float64)
    max_values = np.full(n_ports, -np.inf, dtype=np.float64)

    # Aggregate moments per dataset to avoid creating one giant concatenated array.
    for data in dataset_values:
        data_float = data.astype(np.float64, copy=False)
        sum_values += np.sum(data_float, axis=0)
        sum_squares += np.einsum("ij,ij->j", data_float, data_float, optimize=True)
        min_values = np.minimum(min_values, np.min(data_float, axis=0))
        max_values = np.maximum(max_values, np.max(data_float, axis=0))

    means = sum_values / total_samples
    variances = np.maximum((sum_squares / total_samples) - np.square(means), 0.0)
    std_values = np.sqrt(variances)

    # Use one-based port indexing in outputs to match domain usage (Port 1..N).
    port_ids = np.arange(1, n_ports + 1)

    stats_df = pd.DataFrame(
        {
            "port": port_ids,
            "mean": means,
            "std": std_values,
            "min": min_values,
            "max": max_values,
        }
    )

    # Higher mean is treated as better for ranking because the request asks
    # which ports tend to have the best values.
    stats_df = stats_df.sort_values("mean", ascending=False).reset_index(drop=True)
    stats_df.insert(0, "rank_by_mean", np.arange(1, len(stats_df) + 1))
    print("Statistics table ready.")
    return stats_df


def _write_group_artifacts(
    *,
    group_name: str,
    dataset_values: list[np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    """Compute and save all statistics artifacts for one dataset group.

    Args:
        group_name (str): Human-readable name for the current analysis group.
            This appears in logs and in the manifest CSV. For regime-specific
            analyses, use the regime label. For the aggregated analysis, use a
            label such as ``"global"``.
        dataset_values (list[np.ndarray]): One or more dataset matrices with
            the same number of columns. Passing a single regime array produces
            a per-regime analysis. Passing multiple arrays produces a combined
            global analysis over all supplied samples.
        output_dir (Path): Folder where all artifacts for this group will be
            written. The directory is created if needed.

    Returns:
        dict[str, Any]: Manifest-style metadata describing the generated
            artifacts, sample counts, and output paths for the group.

    Raises:
        ValueError: If the input datasets are inconsistent or empty.
        OSError: If any artifact cannot be written.

    Examples:
        >>> demo = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> result = _write_group_artifacts(
        ...     group_name="demo",
        ...     dataset_values=[demo],
        ...     output_dir=Path("/tmp/demo_ports"),
        ... )
        >>> result["group_name"]
        'demo'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reusing the common validation step keeps the per-regime and global views
    # numerically consistent and prevents silently mixing incompatible arrays.
    group_dataset_names = [f"{group_name}_{index}" for index in range(len(dataset_values))]
    n_ports, total_samples, global_min, global_max = _validate_datasets_and_get_global_range(
        group_dataset_names,
        dataset_values,
    )
    stats_df = _build_port_statistics(dataset_values, n_ports=n_ports, total_samples=total_samples)

    stats_output = output_dir / "port_statistics.csv"
    stats_df.to_csv(stats_output, index=False)

    mean_plot_output = output_dir / "mean_x_ports.png"
    _plot_metric_x_ports(
        stats_df=stats_df,
        metric_column="mean",
        line_color="tab:blue",
        title=f"Mean x ports - {group_name}",
        output_path=mean_plot_output,
    )

    max_plot_output = output_dir / "max_x_ports.png"
    _plot_metric_x_ports(
        stats_df=stats_df,
        metric_column="max",
        line_color="tab:red",
        title=f"Max x ports - {group_name}",
        output_path=max_plot_output,
    )

    std_plot_output = output_dir / "std_x_ports.png"
    _plot_metric_x_ports(
        stats_df=stats_df,
        metric_column="std",
        line_color="tab:green",
        title=f"Std x ports - {group_name}",
        output_path=std_plot_output,
    )

    max_hist_output = output_dir / "max_histogram.png"
    _plot_max_histogram(stats_df, output_path=max_hist_output)

    return {
        "group_name": group_name,
        "output_dir": str(output_dir),
        "n_subsets": int(len(dataset_values)),
        "n_ports": int(n_ports),
        "total_samples": int(total_samples),
        "global_min": float(global_min),
        "global_max": float(global_max),
        "stats_csv": str(stats_output),
        "mean_plot": str(mean_plot_output),
        "max_plot": str(max_plot_output),
        "std_plot": str(std_plot_output),
        "max_histogram": str(max_hist_output),
    }


def _plot_metric_x_ports(
    stats_df: pd.DataFrame,
    metric_column: str,
    line_color: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot one metric against port index and save a dedicated figure.

    Args:
        stats_df (pd.DataFrame): Statistics table containing at least
            ``port`` and the selected metric column.
        metric_column (str): Column name to be plotted on the Y axis.
        line_color (str): Matplotlib color for the metric line.
        title (str): Figure title.
        output_path (Path): Target PNG path.

    Returns:
        None: This function writes the figure to disk.

    Raises:
        ValueError: If required columns are missing in ``stats_df``.

    Examples:
        >>> demo = pd.DataFrame({"port": [1, 2], "mean": [1.0, 2.0]})
        >>> _plot_metric_x_ports(demo, "mean", "tab:blue", "Mean x ports", Path("/tmp/mean.png"))
    """
    print(f"Rendering {metric_column} x ports plot...")

    required_columns = {"port", metric_column}
    missing_columns = required_columns.difference(stats_df.columns)
    if missing_columns:
        raise ValueError(f"stats_df is missing required columns: {sorted(missing_columns)}")

    # Sort by port so X axis is the port index order (Port 1..N).
    ordered = stats_df.sort_values("port", ascending=True)
    x_ports = ordered["port"].to_numpy()
    metric_values = ordered[metric_column].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x_ports, metric_values, linewidth=2.0, color=line_color)
    ax.scatter(x_ports, metric_values, s=12, color=line_color, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Port index")
    ax.set_ylabel(metric_column.upper())
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def _plot_max_histogram(stats_df: pd.DataFrame, output_path: Path) -> None:
    """Plot per-port maximum values with port index on the X axis.

    Args:
        stats_df (pd.DataFrame): Statistics table containing ``port`` and
            ``max`` columns with one row per port.
        output_path (Path): Target PNG path.

    Returns:
        None: This function writes the figure to disk.

    Raises:
        ValueError: If ``port`` or ``max`` is missing.

    Examples:
        >>> demo = pd.DataFrame({"port": [1, 2, 3], "max": [1.0, 2.0, 3.0]})
        >>> _plot_max_histogram(demo, Path("/tmp/max_hist.png"))
    """
    print("Rendering max-only plot with port index on X axis...")

    required_columns = {"port", "max"}
    missing_columns = required_columns.difference(stats_df.columns)
    if missing_columns:
        raise ValueError(f"stats_df is missing required columns: {sorted(missing_columns)}")

    # Keep the canonical port order on X so the chart goes from port 1 to port N.
    ordered = stats_df.sort_values("port", ascending=True)
    port_values = ordered["port"].to_numpy()
    max_values = ordered["max"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(port_values, max_values, color="tab:red", alpha=0.8, width=0.9)
    ax.set_title("Max values by port index")
    ax.set_xlabel("Port index")
    ax.set_ylabel("Max value")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Max-by-port plot saved: {output_path}")


def run_ports_histogram_analysis() -> None:
    """Execute separated Mean/Max/Std plots and max histogram analysis.

    The analysis flow is:
    1) load the selected dataset groups,
    2) create one output folder per regime/dataset,
    3) compute and save all per-port statistics for each regime separately,
    4) compute and save one additional global analysis across all regimes,
    5) write a manifest CSV at the root so every artifact can be found easily.

    Args:
        None: This workflow is configured by module-level global variables.

    Returns:
        None: Results are written to disk and a concise status is printed.

    Raises:
        ValueError: If datasets are inconsistent (for example, different port counts).
        FileNotFoundError: Propagated if dataset files are missing in the selected loader.
        KeyError: Propagated if required keys are absent in MATLAB files.

    Examples:
        >>> run_ports_histogram_analysis()
    """
    print("Starting ports histogram analysis...")
    print(f"Configuration: USE_LOAD_DATA_ALL={USE_LOAD_DATA_ALL}")
    if COMMON_DATA_PATH is None:
        print("Configuration: COMMON_DATA_PATH=None (using loader defaults)")
    else:
        print(f"Configuration: COMMON_DATA_PATH={COMMON_DATA_PATH}")

    dataset_names, dataset_values = _load_selected_datasets(
        use_load_data_all=USE_LOAD_DATA_ALL,
        common_path=COMMON_DATA_PATH,
    )

    # The root output folder stores a manifest and two clean subtrees:
    # one for the regime-specific outputs and another for the global summary.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {OUTPUT_DIR}")
    per_regime_dir = OUTPUT_DIR / "per_regime"
    global_dir = OUTPUT_DIR / "global"
    per_regime_dir.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    # Save each regime separately so it is easy to compare folders side by side
    # without having to manually split a single aggregated output later.
    for dataset_name, dataset_value in zip(dataset_names, dataset_values, strict=True):
        regime_dir = per_regime_dir / _sanitize_group_name(dataset_name)
        print(f"[4/6] Processing regime '{dataset_name}' into {regime_dir}...")
        manifest_row = _write_group_artifacts(
            group_name=dataset_name,
            dataset_values=[dataset_value],
            output_dir=regime_dir,
        )
        manifest_row["scope"] = "per_regime"
        manifest_rows.append(manifest_row)

    # The global analysis intentionally reuses the exact same logic, but now
    # the statistics are computed over the union of all loaded regimes.
    print("[5/6] Processing global analysis across all loaded regimes...")
    global_manifest_row = _write_group_artifacts(
        group_name="global",
        dataset_values=dataset_values,
        output_dir=global_dir,
    )
    global_manifest_row["scope"] = "global"
    manifest_rows.append(global_manifest_row)

    manifest_output = OUTPUT_DIR / "artifact_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_output, index=False)

    print("[6/6] Analysis finished.")
    print(
        "Saved outputs:\n"
        f"- Manifest: {manifest_output}\n"
        f"- Per-regime directory: {per_regime_dir}\n"
        f"- Global directory: {global_dir}\n"
        f"Loader mode: {'load_data_all' if USE_LOAD_DATA_ALL else 'load_data'}\n"
        f"Regimes/datasets processed: {len(dataset_names)}"
    )


if __name__ == "__main__":
    run_ports_histogram_analysis()
