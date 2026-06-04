"""Generate the required Trabalho 04 plots from simulation CSV outputs."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
BASE_SCENARIO_ID = "base"
TIME_SERIES_WINDOW = 120.0
QUEUE_ORDER = [
    "mm1_buffer_inteligente_descarte_seletivo",
    "mm1_buffer_finito_fcfs",
    "mm1_buffer_infinito_fcfs",
    "mg1_buffer_infinito_fcfs",
    "mmm_buffer_infinito_fcfs",
    "prioridade_sem_preempcao",
]
QUEUE_DISPLAY_LABELS = {
    "mm1_buffer_inteligente_descarte_seletivo": "M/M/1 Inteligente",
    "mm1_buffer_finito_fcfs": "M/M/1/K FCFS",
    "mm1_buffer_infinito_fcfs": "M/M/1 inf FCFS",
    "mg1_buffer_infinito_fcfs": "M/G/1 inf FCFS",
    "mmm_buffer_infinito_fcfs": "M/M/m inf FCFS",
    "prioridade_sem_preempcao": "Prioridade sem\nPreempção",
}


def ensure_dir(path):
    """Create a directory path if it does not already exist.

    Args:
        path (str | Path): Directory path to create. Missing parent directories
            are created automatically.

    Returns:
        Path: Resolved directory path.

    Raises:
        OSError: If the directory cannot be created.
    """
    path_obj = Path(path).resolve()
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def configure_plot_style():
    """Apply a restrained Matplotlib style for all generated figures.

    Args:
        None: The function updates Matplotlib global rcParams in place.

    Returns:
        None: Subsequent figures inherit the configured style.

    Raises:
        None: RcParam assignment is deterministic for the local process.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": False,
        }
    )


def get_queue_display_label(queue_slug, queue_name):
    """Return the short display label used only in plot annotations.

    Args:
        queue_slug (str): Stable queue identifier used to select a custom plot
            label when readability would suffer with the full official name.
        queue_name (str): Official queue name stored in the result files. This
            value is preserved as the fallback when no shorter plot label is
            defined for the slug.

    Returns:
        str: Short label intended for axis ticks and legend entries.

    Raises:
        None: Missing slugs simply fall back to `queue_name`.
    """
    return QUEUE_DISPLAY_LABELS.get(queue_slug, queue_name)


def wrap_tick_label(label, max_words_per_line=2):
    """Split a long tick label into multiple lines for readability.

    Args:
        label (str): Tick label text to wrap. Existing newline characters are
            preserved because some labels are already manually shortened.
        max_words_per_line (int): Maximum number of whitespace-delimited words
            placed on each line. Smaller values produce taller labels and less
            horizontal overlap. Larger values keep labels flatter but can
            collide on dense categorical axes.

    Returns:
        str: Wrapped label text with embedded newline characters when needed.

    Raises:
        ValueError: If `max_words_per_line` is smaller than `1`.
    """
    if max_words_per_line < 1:
        raise ValueError("max_words_per_line must be at least 1.")
    if "\n" in label:
        return label

    words = label.split()
    if len(words) <= max_words_per_line:
        return label

    wrapped_lines = []
    for start_index in range(0, len(words), max_words_per_line):
        wrapped_lines.append(" ".join(words[start_index : start_index + max_words_per_line]))
    return "\n".join(wrapped_lines)


def style_axis(axis):
    """Apply restrained scientific styling to one Matplotlib axis.

    Args:
        axis (matplotlib.axes.Axes): Axis object to style. The function keeps
            the plot semantics unchanged and only adjusts readability features
            such as spines and grid lines.

    Returns:
        None: The axis is updated in place.

    Raises:
        None: Axis styling is deterministic.
    """
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    axis.set_axisbelow(True)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)


def read_csv_rows(path):
    """Read all rows from a CSV file into a list of dictionaries.

    Args:
        path (str | Path): CSV file path to read.

    Returns:
        list[dict[str, str]]: Parsed rows with string values.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        OSError: If the file cannot be read.
        csv.Error: If CSV parsing fails.
    """
    with open(Path(path).resolve(), "r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def load_summary_rows():
    """Load every `summary.csv` file produced by the simulation scripts.

    Args:
        None: The function scans the expected `results/*/summary.csv` paths.

    Returns:
        list[dict[str, str]]: Concatenated summary rows across all queue types.

    Raises:
        FileNotFoundError: If no summary files are found.
    """
    summary_rows = []
    for queue_slug in QUEUE_ORDER:
        summary_path = RESULTS_DIR / queue_slug / "summary.csv"
        if summary_path.exists():
            summary_rows.extend(read_csv_rows(summary_path))
    if not summary_rows:
        raise FileNotFoundError("No summary.csv files were found under results/.")
    return summary_rows


def load_class_summary_rows():
    """Load every `class_summary.csv` file produced by the simulation scripts.

    Args:
        None: The function scans the expected `results/*/class_summary.csv`
            paths.

    Returns:
        list[dict[str, str]]: Concatenated class-summary rows across queue
            types.

    Raises:
        FileNotFoundError: If no class-summary files are found.
    """
    class_rows = []
    for queue_slug in QUEUE_ORDER:
        class_path = RESULTS_DIR / queue_slug / "class_summary.csv"
        if class_path.exists():
            class_rows.extend(read_csv_rows(class_path))
    if not class_rows:
        raise FileNotFoundError("No class_summary.csv files were found under results/.")
    return class_rows


def summary_lookup(summary_rows, queue_slug, scenario_id, metric):
    """Find one long-format summary row by queue, scenario, and metric.

    Args:
        summary_rows (list[dict[str, str]]): All summary rows loaded from disk.
        queue_slug (str): Queue identifier to match.
        scenario_id (str): Scenario identifier to match.
        metric (str): Metric name to match.

    Returns:
        dict[str, str]: Matching summary row.

    Raises:
        KeyError: If no matching row exists.
    """
    for row in summary_rows:
        if row["queue_slug"] == queue_slug and row["scenario_id"] == scenario_id and row["metric"] == metric:
            return row
    raise KeyError(f"Missing summary row for {queue_slug=} {scenario_id=} {metric=}.")


def class_summary_lookup(class_rows, queue_slug, scenario_id, priority, metric):
    """Find one class-summary row by queue, scenario, class, and metric.

    Args:
        class_rows (list[dict[str, str]]): All class-summary rows loaded from
            disk.
        queue_slug (str): Queue identifier to match.
        scenario_id (str): Scenario identifier to match.
        priority (str): Priority label, either `"alta"` or `"baixa"`.
        metric (str): Metric name to match.

    Returns:
        dict[str, str]: Matching class-summary row.

    Raises:
        KeyError: If no matching row exists.
    """
    for row in class_rows:
        if (
            row["queue_slug"] == queue_slug
            and row["scenario_id"] == scenario_id
            and row["priority"] == priority
            and row["metric"] == metric
        ):
            return row
    raise KeyError(f"Missing class summary row for {queue_slug=} {scenario_id=} {priority=} {metric=}.")


def plot_bar_comparison(summary_rows, metric, ylabel, filename):
    """Plot one base-scenario bar chart comparing all queue types.

    Args:
        summary_rows (list[dict[str, str]]): Loaded summary rows.
        metric (str): Metric name plotted on the y-axis.
        ylabel (str): Axis label shown in the figure.
        filename (str): Output PNG filename inside `results/plots/`.

    Returns:
        None: The figure is saved to disk.

    Raises:
        KeyError: If any required summary row is missing.
    """
    labels = []
    means = []
    errors = []
    for queue_slug in QUEUE_ORDER:
        row = summary_lookup(summary_rows, queue_slug, BASE_SCENARIO_ID, metric)
        labels.append(wrap_tick_label(get_queue_display_label(queue_slug, row["queue_name"])))
        means.append(float(row["mean"]))
        errors.append(max(0.0, float(row["ci95_high"]) - float(row["mean"])))

    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.bar(labels, means, yerr=errors, capsize=4)
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel} no cenário base")
    axis.tick_params(axis="x", rotation=0)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close(figure)


def plot_priority_bars(class_rows, metric, ylabel, filename):
    """Plot grouped class-metric bars for the base scenario across queues.

    Args:
        class_rows (list[dict[str, str]]): Loaded class-summary rows.
        metric (str): Class metric name plotted on the y-axis.
        ylabel (str): Axis label shown in the figure.
        filename (str): Output PNG filename inside `results/plots/`.

    Returns:
        None: The figure is saved to disk.

    Raises:
        KeyError: If any required class-summary row is missing.
    """
    labels = []
    high_means = []
    low_means = []
    positions = list(range(len(QUEUE_ORDER)))
    for queue_slug in QUEUE_ORDER:
        high_row = class_summary_lookup(class_rows, queue_slug, BASE_SCENARIO_ID, "alta", metric)
        low_row = class_summary_lookup(class_rows, queue_slug, BASE_SCENARIO_ID, "baixa", metric)
        labels.append(wrap_tick_label(get_queue_display_label(queue_slug, high_row["queue_name"])))
        high_means.append(float(high_row["mean"]))
        low_means.append(float(low_row["mean"]))

    figure, axis = plt.subplots(figsize=(10, 5.8))
    width = 0.4
    axis.bar([position - (width / 2.0) for position in positions], high_means, width=width, label="alta")
    axis.bar([position + (width / 2.0) for position in positions], low_means, width=width, label="baixa")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=0)
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel} por prioridade no cenário base")
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close(figure)


def plot_buffer_effect(summary_rows):
    """Plot how finite-buffer size changes total loss probability.

    Args:
        summary_rows (list[dict[str, str]]): Loaded summary rows.

    Returns:
        None: The figure is saved to disk.

    Raises:
        KeyError: If the expected summary rows are missing.
    """
    scenarios = [("buffer_K_2", 2), ("base", 5), ("buffer_K_10", 10), ("buffer_K_20", 20)]
    figure, axis = plt.subplots(figsize=(8, 5.2))
    for queue_slug in ("mm1_buffer_finito_fcfs", "mm1_buffer_inteligente_descarte_seletivo"):
        means = []
        errors = []
        buffer_sizes = []
        for scenario_id, buffer_size in scenarios:
            row = summary_lookup(summary_rows, queue_slug, scenario_id, "loss_probability_total")
            buffer_sizes.append(buffer_size)
            means.append(float(row["mean"]))
            errors.append(max(0.0, float(row["ci95_high"]) - float(row["mean"])))
        axis.errorbar(
            buffer_sizes,
            means,
            yerr=errors,
            marker="o",
            capsize=4,
            label=get_queue_display_label(queue_slug, queue_slug),
        )

    axis.set_xlabel("K")
    axis.set_ylabel("Probabilidade de perda total")
    axis.set_title("Efeito do buffer K")
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "efeito_do_buffer_K.png", dpi=200)
    plt.close(figure)


def plot_rho_effect(summary_rows):
    """Plot how utilization changes mean system time across queue models.

    Args:
        summary_rows (list[dict[str, str]]): Loaded summary rows.

    Returns:
        None: The figure is saved to disk.

    Raises:
        KeyError: If the expected summary rows are missing.
    """
    scenario_ids = ["rho_0_50", "rho_0_70", "base", "rho_0_90", "rho_0_95"]
    rho_values = [0.50, 0.70, 0.80, 0.90, 0.95]
    figure, axis = plt.subplots(figsize=(9, 5.2))
    rho_queue_slugs = [
        "mm1_buffer_inteligente_descarte_seletivo",
        "mm1_buffer_finito_fcfs",
        "mm1_buffer_infinito_fcfs",
        "mmm_buffer_infinito_fcfs",
        "prioridade_sem_preempcao",
    ]
    for queue_slug in rho_queue_slugs:
        means = []
        for scenario_id in scenario_ids:
            row = summary_lookup(summary_rows, queue_slug, scenario_id, "mean_system_time")
            means.append(float(row["mean"]))
        axis.plot(rho_values, means, marker="o", label=get_queue_display_label(queue_slug, queue_slug).replace("\n", " "))

    axis.set_xlabel("ρ")
    axis.set_ylabel("Tempo médio no sistema (s)")
    axis.set_title("Efeito da utilização ρ")
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "efeito_da_utilizacao_rho.png", dpi=200)
    plt.close(figure)


def plot_mg1_service_effect(summary_rows):
    """Plot the M/G/1 delay comparison across service distributions.

    Args:
        summary_rows (list[dict[str, str]]): Loaded summary rows.

    Returns:
        None: The figure is saved to disk.

    Raises:
        KeyError: If the expected M/G/1 summary rows are missing.
    """
    labels = ["exponencial", "determinístico", "uniforme"]
    scenario_ids = ["base", "service_deterministic", "service_uniform"]
    means = []
    errors = []
    for scenario_id in scenario_ids:
        row = summary_lookup(summary_rows, "mg1_buffer_infinito_fcfs", scenario_id, "mean_system_time")
        means.append(float(row["mean"]))
        errors.append(max(0.0, float(row["ci95_high"]) - float(row["mean"])))

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(labels, means, yerr=errors, capsize=4)
    axis.set_ylabel("Tempo médio no sistema (s)")
    axis.set_title("Efeito da distribuição de serviço na M/G/1")
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "efeito_da_distribuicao_servico_mg1.png", dpi=200)
    plt.close(figure)


def plot_main_queue_time_series():
    """Plot the base-case time series of the main queue occupancy.

    Args:
        None: The function reads the main queue time-series CSV from disk.

    Returns:
        None: The figure is saved to disk.

    Raises:
        FileNotFoundError: If the required time-series CSV does not exist.
    """
    time_series_path = RESULTS_DIR / "mm1_buffer_inteligente_descarte_seletivo" / "time_series_sample.csv"
    rows = read_csv_rows(time_series_path)
    times = [float(row["time"]) for row in rows]
    num_in_system = [float(row["num_in_system"]) for row in rows]
    num_in_queue = [float(row["num_in_queue"]) for row in rows]

    # The full 900-second measured horizon is too dense for a categorical
    # occupancy trace sampled every second, so the figure focuses on the first
    # representative post-warmup window.
    window_start = times[0]
    window_end = window_start + TIME_SERIES_WINDOW
    filtered_points = [
        (time_value, system_value, queue_value)
        for time_value, system_value, queue_value in zip(times, num_in_system, num_in_queue)
        if time_value <= window_end
    ]
    plot_times = [point[0] for point in filtered_points]
    plot_system = [point[1] for point in filtered_points]
    plot_queue = [point[2] for point in filtered_points]

    figure, axis = plt.subplots(figsize=(10, 5.2))

    # The sampled occupancy is piecewise constant, so a step plot preserves the
    # event-driven semantics and avoids the misleading spikes of linear
    # interpolation between adjacent one-second samples.
    axis.fill_between(
        plot_times,
        plot_queue,
        step="post",
        alpha=0.25,
        color="C1",
        label="Pacotes na fila",
    )
    axis.step(plot_times, plot_queue, where="post", linewidth=1.0, color="C1")
    axis.step(plot_times, plot_system, where="post", linewidth=1.6, color="C0", label="Pacotes no sistema")
    axis.set_xlabel("Tempo (s)")
    axis.set_ylabel("Pacotes")
    axis.set_title(f"Série temporal da ocupação da fila principal ({window_start:.0f} s a {window_end:.0f} s)")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "serie_temporal_ocupacao_fila_principal.png", dpi=200)
    plt.close(figure)


def main():
    """Load simulation summaries and generate every required Trabalho 04 plot.

    Args:
        None: The script reads result CSV files from the local `results/`
            directory and writes PNG files to `results/plots/`.

    Returns:
        None: Plot files are written to disk and a completion summary is
            printed.

    Raises:
        FileNotFoundError: If the simulation outputs do not exist yet.
        KeyError: If an expected metric row is missing from the summaries.
    """
    configure_plot_style()
    ensure_dir(PLOTS_DIR)
    summary_rows = load_summary_rows()
    class_rows = load_class_summary_rows()

    plot_bar_comparison(summary_rows, "loss_probability_total", "Probabilidade de perda total", "comparacao_perda_total_base.png")
    plot_bar_comparison(summary_rows, "mean_system_time", "Tempo médio no sistema (s)", "comparacao_atraso_medio_base.png")
    plot_bar_comparison(summary_rows, "mean_system_occupancy", "Ocupação média do sistema", "comparacao_ocupacao_media_base.png")
    plot_bar_comparison(summary_rows, "server_utilization", "Utilização média dos servidores", "comparacao_utilizacao_base.png")
    plot_bar_comparison(summary_rows, "throughput_total", "Vazão total (pacotes/s)", "comparacao_vazao_base.png")
    plot_priority_bars(class_rows, "loss_probability", "Probabilidade de perda", "perda_por_prioridade_base.png")
    plot_priority_bars(class_rows, "mean_system_time", "Tempo médio no sistema (s)", "atraso_por_prioridade_base.png")
    plot_buffer_effect(summary_rows)
    plot_rho_effect(summary_rows)
    plot_mg1_service_effect(summary_rows)
    plot_main_queue_time_series()

    print("Finalizado: gráficos do Trabalho 04")
    print(f"Resultados salvos em: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
