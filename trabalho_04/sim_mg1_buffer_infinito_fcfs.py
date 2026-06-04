"""Simulate the M/G/1 queue with infinite FCFS buffer for Trabalho 04."""

import heapq
import random
from collections import defaultdict, deque
from pathlib import Path

from common import (
    HIGH_PRIORITY,
    CLASS_SUMMARY_FIELDNAMES,
    OCCUPANCY_FIELDNAMES,
    Packet,
    REPLICATION_FIELDNAMES,
    SUMMARY_FIELDNAMES,
    TIME_SERIES_FIELDNAMES,
    build_admitted_little_metrics,
    collect_class_summary,
    compute_little_law_error,
    ensure_dir,
    exponential_time,
    generate_priority,
    generate_service_time,
    mean,
    percentile,
    safe_divide,
    schedule_event,
    summarize_replications,
    update_area_metrics,
    write_dicts_csv,
    write_json,
)


QUEUE_NAME = "Fila M/G/1/∞/∞/∞/FCFS"
QUEUE_SLUG = "mg1_buffer_infinito_fcfs"
LAMBDA_BASE = 200.0
MU_BASE = 250.0
RHO_BASE = 0.8
K_BASE = ""
P_HIGH_BASE = 0.3
SERVERS_BASE = 1
SIM_TIME = 1000.0
WARMUP_TIME = 100.0
N_REPLICATIONS = 30
BASE_SEED = 5472026
TIME_SERIES_DT = 1.0
SUMMARY_METRIC_NAMES = (
    "lambda_rate",
    "mu_rate",
    "rho",
    "p_high",
    "servers",
    "sim_time",
    "warmup_time",
    "measured_time",
    "arrivals_total",
    "arrivals_high",
    "arrivals_low",
    "accepted_total",
    "accepted_high",
    "accepted_low",
    "served_total",
    "served_high",
    "served_low",
    "dropped_total",
    "dropped_high",
    "dropped_low",
    "replaced_low",
    "loss_probability_total",
    "loss_probability_high",
    "loss_probability_low",
    "replacement_probability_low",
    "throughput_total",
    "throughput_high",
    "throughput_low",
    "server_utilization",
    "mean_system_occupancy",
    "mean_queue_occupancy",
    "mean_waiting_time",
    "mean_system_time",
    "mean_service_time",
    "p50_system_time",
    "p95_system_time",
    "p99_system_time",
    "mean_waiting_time_high",
    "mean_waiting_time_low",
    "mean_system_time_high",
    "mean_system_time_low",
    "little_law_error_abs",
    "little_law_error_rel",
    "mean_residence_time_admitted",
    "departed_admitted_total",
    "admitted_departure_rate",
    "little_law_error_admitted",
)


def build_scenarios():
    """Create the service-distribution scenarios required for M/G/1.

    Args:
        None: The function uses the fixed values from the project plan.

    Returns:
        list[dict]: Scenario dictionaries with `service_distribution` set to
            the required mode.

    Raises:
        None: Scenario construction is deterministic.
    """
    return [
        {"scenario_id": "base", "lambda_rate": 200.0, "mu_rate": 250.0, "rho": 0.80, "K": "", "p_high": 0.30, "service_distribution": "exponential"},
        {"scenario_id": "service_deterministic", "lambda_rate": 200.0, "mu_rate": 250.0, "rho": 0.80, "K": "", "p_high": 0.30, "service_distribution": "deterministic"},
        {"scenario_id": "service_uniform", "lambda_rate": 200.0, "mu_rate": 250.0, "rho": 0.80, "K": "", "p_high": 0.30, "service_distribution": "uniform"},
    ]


def append_time_series_until(sample_rows, next_sample_time, upper_time, scenario_id, replication, num_in_system, num_in_queue, busy_servers):
    """Sample the piecewise-constant system state at one-second intervals.

    Args:
        sample_rows (list[dict]): Mutable collector for output rows.
        next_sample_time (float): Next sample time in seconds.
        upper_time (float): Inclusive upper bound for emitted samples.
        scenario_id (str): Scenario identifier copied into each row.
        replication (int): Replication index copied into each row.
        num_in_system (int): Current number of packets in the system.
        num_in_queue (int): Current number of packets waiting in queue.
        busy_servers (int): Current busy-server count.

    Returns:
        float: Updated next sample timestamp.

    Raises:
        None: Sampling is deterministic and in-memory.
    """
    while next_sample_time <= upper_time:
        sample_rows.append(
            {
                "queue_name": QUEUE_NAME,
                "queue_slug": QUEUE_SLUG,
                "scenario_id": scenario_id,
                "replication": replication,
                "time": next_sample_time,
                "num_in_system": num_in_system,
                "num_in_queue": num_in_queue,
                "busy_servers": busy_servers,
            }
        )
        next_sample_time += TIME_SERIES_DT
    return next_sample_time


def start_service(packet, clock, rng, scenario, event_heap, event_counter):
    """Assign service to one packet using the scenario-selected distribution.

    Args:
        packet (Packet): Packet entering service.
        clock (float): Current simulation time in seconds.
        rng (random.Random): Random generator used for service times.
        scenario (dict): Scenario parameters including `mu_rate` and
            `service_distribution`.
        event_heap (list[tuple]): Pending event heap.
        event_counter (int): Current event tie-breaker counter.

    Returns:
        int: Updated event counter after scheduling the departure.

    Raises:
        ValueError: If the selected service distribution is invalid.
    """
    packet.was_accepted = True
    packet.service_start_time = clock
    packet.service_time = generate_service_time(scenario["mu_rate"], scenario["service_distribution"], rng)
    packet.departure_time = clock + packet.service_time
    return schedule_event(
        event_heap,
        packet.departure_time,
        event_counter,
        "departure",
        packet_id=packet.id,
        server_id=0,
    )


def build_replication_row(scenario, replication, seed, counts, queue_areas, waiting_times, system_times, service_times, waiting_times_by_class, system_times_by_class):
    """Convert one M/G/1 replication into the required CSV row.

    Args:
        scenario (dict): Scenario parameters for the replication.
        replication (int): Zero-based replication index.
        seed (int): Replication seed used to initialize randomness.
        counts (dict[str, int]): Measured count totals.
        queue_areas (dict[str, float]): Time-integrated occupancy metrics.
        waiting_times (list[float]): Warm-up-filtered waiting times.
        system_times (list[float]): Warm-up-filtered system times.
        service_times (list[float]): Service times for eligible completions.
        waiting_times_by_class (dict[str, list[float]]): Waiting times grouped
            by priority.
        system_times_by_class (dict[str, list[float]]): System times grouped by
            priority.

    Returns:
        dict: Replication row aligned to `REPLICATION_FIELDNAMES`.

    Raises:
        ValueError: If percentile computation receives malformed data.
    """
    measured_time = SIM_TIME - WARMUP_TIME
    throughput_total = safe_divide(counts["served_total"], measured_time)
    throughput_high = safe_divide(counts["served_high"], measured_time)
    throughput_low = safe_divide(counts["served_low"], measured_time)
    mean_system_occupancy = safe_divide(queue_areas["system_occupancy_area"], measured_time)
    mean_queue_occupancy = safe_divide(queue_areas["queue_occupancy_area"], measured_time)
    server_utilization = safe_divide(queue_areas["busy_server_area"], measured_time)
    mean_waiting_time = mean(waiting_times) if waiting_times else 0.0
    mean_system_time = mean(system_times) if system_times else 0.0
    mean_service_time = mean(service_times) if service_times else 0.0
    p50_system_time = percentile(system_times, 50.0) if system_times else 0.0
    p95_system_time = percentile(system_times, 95.0) if system_times else 0.0
    p99_system_time = percentile(system_times, 99.0) if system_times else 0.0
    mean_waiting_time_high = mean(waiting_times_by_class["alta"]) if waiting_times_by_class["alta"] else 0.0
    mean_waiting_time_low = mean(waiting_times_by_class["baixa"]) if waiting_times_by_class["baixa"] else 0.0
    mean_system_time_high = mean(system_times_by_class["alta"]) if system_times_by_class["alta"] else 0.0
    mean_system_time_low = mean(system_times_by_class["baixa"]) if system_times_by_class["baixa"] else 0.0
    little_law_error_abs, little_law_error_rel = compute_little_law_error(
        mean_system_occupancy,
        throughput_total,
        mean_system_time,
    )
    admitted_little_metrics = build_admitted_little_metrics(
        mean_system_occupancy,
        measured_time,
        counts["served_total"],
        0,
        system_times,
        [],
    )

    return {
        "queue_name": QUEUE_NAME,
        "queue_slug": QUEUE_SLUG,
        "scenario_id": scenario["scenario_id"],
        "replication": replication,
        "seed": seed,
        "lambda_rate": scenario["lambda_rate"],
        "mu_rate": scenario["mu_rate"],
        "rho": scenario["rho"],
        "K": scenario["K"],
        "N": "",
        "p_high": scenario["p_high"],
        "servers": 1,
        "service_distribution": scenario["service_distribution"],
        "sim_time": SIM_TIME,
        "warmup_time": WARMUP_TIME,
        "measured_time": measured_time,
        "arrivals_total": counts["arrivals_total"],
        "arrivals_high": counts["arrivals_high"],
        "arrivals_low": counts["arrivals_low"],
        "accepted_total": counts["accepted_total"],
        "accepted_high": counts["accepted_high"],
        "accepted_low": counts["accepted_low"],
        "served_total": counts["served_total"],
        "served_high": counts["served_high"],
        "served_low": counts["served_low"],
        "dropped_total": 0,
        "dropped_high": 0,
        "dropped_low": 0,
        "replaced_low": 0,
        "loss_probability_total": 0.0,
        "loss_probability_high": 0.0,
        "loss_probability_low": 0.0,
        "replacement_probability_low": 0.0,
        "throughput_total": throughput_total,
        "throughput_high": throughput_high,
        "throughput_low": throughput_low,
        "server_utilization": server_utilization,
        "mean_system_occupancy": mean_system_occupancy,
        "mean_queue_occupancy": mean_queue_occupancy,
        "mean_waiting_time": mean_waiting_time,
        "mean_system_time": mean_system_time,
        "mean_service_time": mean_service_time,
        "p50_system_time": p50_system_time,
        "p95_system_time": p95_system_time,
        "p99_system_time": p99_system_time,
        "mean_waiting_time_high": mean_waiting_time_high,
        "mean_waiting_time_low": mean_waiting_time_low,
        "mean_system_time_high": mean_system_time_high,
        "mean_system_time_low": mean_system_time_low,
        "little_law_error_abs": little_law_error_abs,
        "little_law_error_rel": little_law_error_rel,
        "mean_residence_time_admitted": admitted_little_metrics["mean_residence_time_admitted"],
        "departed_admitted_total": admitted_little_metrics["departed_admitted_total"],
        "admitted_departure_rate": admitted_little_metrics["admitted_departure_rate"],
        "little_law_error_admitted": admitted_little_metrics["little_law_error_admitted"],
    }


def simulate_replication(scenario, replication):
    """Run one event-driven replication of the infinite-buffer M/G/1 queue.

    Args:
        scenario (dict): Scenario parameters with traffic and service mode.
        replication (int): Zero-based replication index.

    Returns:
        tuple[dict, list[dict], list[dict]]: Replication row, occupancy rows,
            and optional time-series rows for the base scenario.

    Raises:
        RuntimeError: If a departure event does not match the packet in
            service.
    """
    seed = BASE_SEED + (1000 * replication) + len(scenario["service_distribution"])
    rng = random.Random(seed)
    event_heap = []
    event_counter = 0
    packet_sequence = 0
    event_counter = schedule_event(event_heap, exponential_time(scenario["lambda_rate"], rng), event_counter, "arrival")

    queue = deque()
    current_packet = None
    last_event_time = 0.0
    next_sample_time = WARMUP_TIME
    occupancy_state_times = defaultdict(float)
    queue_areas = {
        "system_occupancy_area": 0.0,
        "queue_occupancy_area": 0.0,
        "busy_server_area": 0.0,
    }
    counts = defaultdict(int)
    waiting_times = []
    system_times = []
    service_times = []
    waiting_times_by_class = defaultdict(list)
    system_times_by_class = defaultdict(list)
    sample_rows = []

    while event_heap:
        event_time, _, event_type, packet_id, _ = heapq.heappop(event_heap)
        effective_time = min(event_time, SIM_TIME)
        num_in_queue = len(queue)
        num_in_system = num_in_queue + (1 if current_packet is not None else 0)
        busy_servers = 1 if current_packet is not None else 0

        queue_areas["system_occupancy_area"], queue_areas["queue_occupancy_area"], queue_areas["busy_server_area"] = update_area_metrics(
            last_time=last_event_time,
            current_time=effective_time,
            warmup_time=WARMUP_TIME,
            sim_time=SIM_TIME,
            num_in_system=num_in_system,
            num_in_queue=num_in_queue,
            busy_servers=busy_servers,
            occupancy_state_times=occupancy_state_times,
            system_occupancy_area=queue_areas["system_occupancy_area"],
            queue_occupancy_area=queue_areas["queue_occupancy_area"],
            busy_server_area=queue_areas["busy_server_area"],
        )
        if scenario["scenario_id"] == "base" and replication == 0:
            next_sample_time = append_time_series_until(
                sample_rows,
                next_sample_time,
                effective_time,
                scenario["scenario_id"],
                replication,
                num_in_system,
                num_in_queue,
                busy_servers,
            )
        last_event_time = effective_time
        if event_time > SIM_TIME:
            break

        if event_type == "arrival":
            packet = Packet(
                id=packet_sequence,
                priority=generate_priority(scenario["p_high"], rng),
                arrival_time=event_time,
                service_start_time=None,
                departure_time=None,
                service_time=None,
                drop_time=None,
                drop_reason=None,
                was_accepted=False,
                was_dropped=False,
                was_replaced=False,
            )
            packet_sequence += 1
            if event_time >= WARMUP_TIME:
                counts["arrivals_total"] += 1
                if packet.priority == HIGH_PRIORITY:
                    counts["arrivals_high"] += 1
                else:
                    counts["arrivals_low"] += 1

            if current_packet is None:
                current_packet = packet
                event_counter = start_service(packet, event_time, rng, scenario, event_heap, event_counter)
            else:
                queue.append(packet)

            if event_time >= WARMUP_TIME:
                counts["accepted_total"] += 1
                if packet.priority == HIGH_PRIORITY:
                    counts["accepted_high"] += 1
                else:
                    counts["accepted_low"] += 1

            next_arrival_time = event_time + exponential_time(scenario["lambda_rate"], rng)
            event_counter = schedule_event(event_heap, next_arrival_time, event_counter, "arrival")
        else:
            if current_packet is None or current_packet.id != packet_id:
                raise RuntimeError("Departure event does not match the packet in service.")

            finished_packet = current_packet
            finished_packet.departure_time = event_time
            current_packet = None

            if finished_packet.arrival_time >= WARMUP_TIME:
                counts["served_total"] += 1
                if finished_packet.priority == HIGH_PRIORITY:
                    counts["served_high"] += 1
                else:
                    counts["served_low"] += 1

            if finished_packet.arrival_time >= WARMUP_TIME:
                waiting_time = finished_packet.service_start_time - finished_packet.arrival_time
                system_time = finished_packet.departure_time - finished_packet.arrival_time
                service_time = finished_packet.service_time
                waiting_times.append(waiting_time)
                system_times.append(system_time)
                service_times.append(service_time)
                waiting_times_by_class[finished_packet.priority].append(waiting_time)
                system_times_by_class[finished_packet.priority].append(system_time)

            if queue:
                current_packet = queue.popleft()
                event_counter = start_service(current_packet, event_time, rng, scenario, event_heap, event_counter)

    if scenario["scenario_id"] == "base" and replication == 0:
        num_in_queue = len(queue)
        num_in_system = num_in_queue + (1 if current_packet is not None else 0)
        busy_servers = 1 if current_packet is not None else 0
        append_time_series_until(
            sample_rows,
            next_sample_time,
            SIM_TIME,
            scenario["scenario_id"],
            replication,
            num_in_system,
            num_in_queue,
            busy_servers,
        )

    replication_row = build_replication_row(
        scenario,
        replication,
        seed,
        counts,
        queue_areas,
        waiting_times,
        system_times,
        service_times,
        waiting_times_by_class,
        system_times_by_class,
    )
    occupancy_rows = []
    measured_time = SIM_TIME - WARMUP_TIME
    for state in sorted(occupancy_state_times):
        occupancy_rows.append(
            {
                "queue_name": QUEUE_NAME,
                "queue_slug": QUEUE_SLUG,
                "scenario_id": scenario["scenario_id"],
                "replication": replication,
                "state": state,
                "time_in_state": occupancy_state_times[state],
                "probability": safe_divide(occupancy_state_times[state], measured_time),
            }
        )
    return replication_row, occupancy_rows, sample_rows


def save_outputs(replication_rows, occupancy_rows, sample_rows, scenarios):
    """Write all required result files for the M/G/1 queue.

    Args:
        replication_rows (list[dict]): Replication-level simulation outputs.
        occupancy_rows (list[dict]): Occupancy-state outputs.
        sample_rows (list[dict]): Time-series rows for the base scenario.
        scenarios (list[dict]): Scenario definitions for `config.json`.

    Returns:
        Path: Output directory used by the script.

    Raises:
        OSError: If any result file cannot be written.
    """
    results_dir = ensure_dir(Path(__file__).resolve().parent / "results" / QUEUE_SLUG)
    config = {
        "queue_name": QUEUE_NAME,
        "queue_slug": QUEUE_SLUG,
        "sim_time": SIM_TIME,
        "warmup_time": WARMUP_TIME,
        "n_replications": N_REPLICATIONS,
        "base_seed": BASE_SEED,
        "scenarios": scenarios,
        "description": "Fila M/G/1 com buffer infinito, chegadas Poisson, disciplina FCFS e distribuicoes de servico exponencial, deterministica e uniforme.",
    }
    write_json(results_dir / "config.json", config)
    write_dicts_csv(results_dir / "replications.csv", replication_rows, REPLICATION_FIELDNAMES)
    write_dicts_csv(
        results_dir / "summary.csv",
        summarize_replications(replication_rows, SUMMARY_METRIC_NAMES),
        SUMMARY_FIELDNAMES,
    )
    write_dicts_csv(
        results_dir / "class_summary.csv",
        collect_class_summary(replication_rows),
        CLASS_SUMMARY_FIELDNAMES,
    )
    write_dicts_csv(results_dir / "occupancy_states.csv", occupancy_rows, OCCUPANCY_FIELDNAMES)
    write_dicts_csv(results_dir / "time_series_sample.csv", sample_rows, TIME_SERIES_FIELDNAMES)
    return results_dir


def main():
    """Run every scenario and replication for the infinite-buffer M/G/1 queue.

    Args:
        None: The script uses the module-level constants and scenario builder.

    Returns:
        None: Results are written to disk and completion is printed.

    Raises:
        RuntimeError: If a departure event references the wrong packet.
        OSError: If output files cannot be written.
    """
    scenarios = build_scenarios()
    replication_rows = []
    occupancy_rows = []
    sample_rows = []

    for scenario in scenarios:
        for replication in range(N_REPLICATIONS):
            row, state_rows, series_rows = simulate_replication(scenario, replication)
            replication_rows.append(row)
            occupancy_rows.extend(state_rows)
            if series_rows:
                sample_rows = series_rows

    results_dir = save_outputs(replication_rows, occupancy_rows, sample_rows, scenarios)
    print(f"Finalizado: {QUEUE_NAME.replace('∞', 'inf')}")
    print(f"Resultados salvos em: {results_dir}")
    print(f"Replicações: {len(replication_rows)}")


if __name__ == "__main__":
    main()
