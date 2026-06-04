"""Simulate the intelligent selective-drop M/M/1 queue for Trabalho 04."""

import heapq
import random
from collections import defaultdict
from pathlib import Path

from common import (
    DROP_REASON_LOW_FULL,
    DROP_REASON_NO_LOW,
    DROP_REASON_REPLACED,
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


QUEUE_NAME = "Fila M/M/1 com Buffer Inteligente e Descarte Seletivo"
QUEUE_SLUG = "mm1_buffer_inteligente_descarte_seletivo"
LAMBDA_BASE = 200.0
MU_BASE = 250.0
RHO_BASE = 0.8
K_BASE = 5
P_HIGH_BASE = 0.3
SERVERS_BASE = 1
SIM_TIME = 1000.0
WARMUP_TIME = 100.0
N_REPLICATIONS = 30
BASE_SEED = 5472026
TIME_SERIES_DT = 1.0
SERVICE_DISTRIBUTION = "exponential"
SUMMARY_METRIC_NAMES = (
    "lambda_rate",
    "mu_rate",
    "rho",
    "K",
    "N",
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
    """Create the full scenario list for the intelligent-buffer queue.

    Args:
        None: The function uses the fixed values mandated by the project plan.

    Returns:
        list[dict]: Scenario dictionaries for common, finite-buffer, and
            priority-variation cases.

    Raises:
        None: Scenario construction is deterministic.
    """
    return [
        {"scenario_id": "base", "lambda_rate": 200.0, "mu_rate": 250.0, "rho": 0.80, "K": 5, "p_high": 0.30},
        {"scenario_id": "rho_0_50", "lambda_rate": 125.0, "mu_rate": 250.0, "rho": 0.50, "K": 5, "p_high": 0.30},
        {"scenario_id": "rho_0_70", "lambda_rate": 175.0, "mu_rate": 250.0, "rho": 0.70, "K": 5, "p_high": 0.30},
        {"scenario_id": "rho_0_90", "lambda_rate": 225.0, "mu_rate": 250.0, "rho": 0.90, "K": 5, "p_high": 0.30},
        {"scenario_id": "rho_0_95", "lambda_rate": 237.5, "mu_rate": 250.0, "rho": 0.95, "K": 5, "p_high": 0.30},
        {"scenario_id": "buffer_K_2", "lambda_rate": 200.0, "mu_rate": 250.0, "rho": 0.80, "K": 2, "p_high": 0.30},
        {"scenario_id": "buffer_K_10", "lambda_rate": 200.0, "mu_rate": 250.0, "rho": 0.80, "K": 10, "p_high": 0.30},
        {"scenario_id": "buffer_K_20", "lambda_rate": 200.0, "mu_rate": 250.0, "rho": 0.80, "K": 20, "p_high": 0.30},
        {"scenario_id": "phigh_0_10", "lambda_rate": 200.0, "mu_rate": 250.0, "rho": 0.80, "K": 5, "p_high": 0.10},
        {"scenario_id": "phigh_0_50", "lambda_rate": 200.0, "mu_rate": 250.0, "rho": 0.80, "K": 5, "p_high": 0.50},
        {"scenario_id": "phigh_0_80", "lambda_rate": 200.0, "mu_rate": 250.0, "rho": 0.80, "K": 5, "p_high": 0.80},
    ]


def append_time_series_until(sample_rows, next_sample_time, upper_time, scenario_id, replication, num_in_system, num_in_queue, busy_servers):
    """Sample the piecewise-constant queue state at fixed one-second intervals.

    Args:
        sample_rows (list[dict]): Mutable collector for the time-series output.
        next_sample_time (float): Next sample time in seconds.
        upper_time (float): Inclusive upper bound for emitted samples.
        scenario_id (str): Scenario identifier copied into each row.
        replication (int): Replication index copied into each row.
        num_in_system (int): Current system occupancy.
        num_in_queue (int): Current queue occupancy.
        busy_servers (int): Busy-server count during the interval.

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


def start_service(packet, clock, rng, mu_rate, event_heap, event_counter):
    """Assign service to one packet and schedule its departure event.

    Args:
        packet (Packet): Packet entering service.
        clock (float): Current simulation time in seconds.
        rng (random.Random): Random generator used for service times.
        mu_rate (float): Service rate in packets per second.
        event_heap (list[tuple]): Pending event heap.
        event_counter (int): Current monotonic event counter.

    Returns:
        int: Updated event counter after scheduling the departure.

    Raises:
        ValueError: If the service distribution arguments are invalid.
    """
    packet.was_accepted = True
    packet.service_start_time = clock
    packet.service_time = generate_service_time(mu_rate, SERVICE_DISTRIBUTION, rng)
    packet.departure_time = clock + packet.service_time
    return schedule_event(
        event_heap,
        packet.departure_time,
        event_counter,
        "departure",
        packet_id=packet.id,
        server_id=0,
    )


def register_drop(packet, drop_time, drop_reason, counts):
    """Register one dropped packet in the packet state and measured counters.

    Args:
        packet (Packet): Packet being discarded.
        drop_time (float): Simulation clock time in seconds when the drop
            occurs.
        drop_reason (str): Drop reason slug required by the specification.
        counts (dict[str, int]): Mutable measured count dictionary. Drop counts
            are recorded only for packets whose arrival belongs to the measured
            warm-up-filtered cohort.

    Returns:
        None: The function mutates `packet` and `counts` in place.

    Raises:
        ValueError: If the packet was already marked as dropped.
    """
    if packet.was_dropped:
        raise ValueError("A packet cannot be dropped twice.")
    packet.was_dropped = True
    packet.drop_time = drop_time
    packet.drop_reason = drop_reason
    if packet.arrival_time >= WARMUP_TIME:
        counts["dropped_total"] += 1
        if packet.priority == HIGH_PRIORITY:
            counts["dropped_high"] += 1
        else:
            counts["dropped_low"] += 1


def update_replaced_low_count(counts, reference_time):
    """Count a low-priority replacement only inside the measured cohort.

    Args:
        counts (dict[str, int]): Mutable measured count dictionary.
        reference_time (float): Warm-up cohort reference time in seconds. The
            main queue uses the replaced packet arrival time so the numerator
            matches the low-arrival denominator in `replacement_probability_low`.

    Returns:
        None: The function mutates `counts` in place.

    Raises:
        KeyError: If `counts` does not contain the `replaced_low` key.
    """
    if reference_time >= WARMUP_TIME:
        counts["replaced_low"] += 1


def build_replication_row(
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
    replaced_residence_times,
):
    """Convert one intelligent-buffer replication into the required CSV row.

    Args:
        scenario (dict): Scenario parameters for the replication.
        replication (int): Zero-based replication index.
        seed (int): Replication seed used to initialize randomness.
        counts (dict[str, int]): Measured counts for arrivals, admissions,
            completions, drops, and replacements.
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
        counts["replaced_low"],
        system_times,
        replaced_residence_times,
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
        "N": scenario["K"] + 1,
        "p_high": scenario["p_high"],
        "servers": 1,
        "service_distribution": SERVICE_DISTRIBUTION,
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
        "dropped_total": counts["dropped_total"],
        "dropped_high": counts["dropped_high"],
        "dropped_low": counts["dropped_low"],
        "replaced_low": counts["replaced_low"],
        "loss_probability_total": safe_divide(counts["dropped_total"], counts["arrivals_total"]),
        "loss_probability_high": safe_divide(counts["dropped_high"], counts["arrivals_high"]),
        "loss_probability_low": safe_divide(counts["dropped_low"], counts["arrivals_low"]),
        "replacement_probability_low": safe_divide(counts["replaced_low"], counts["arrivals_low"]),
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
    """Run one event-driven replication of the intelligent finite-buffer queue.

    Args:
        scenario (dict): Scenario parameters with traffic, service, buffer, and
            priority-mix values.
        replication (int): Zero-based replication index.

    Returns:
        tuple[dict, list[dict], list[dict]]: Replication row, occupancy rows,
            and optional time-series rows for the base scenario.

    Raises:
        RuntimeError: If a departure event does not match the packet in
            service, if the queue exceeds `K`, or if replacement tries to touch
            the packet in service.
    """
    seed = BASE_SEED + (1000 * replication) + int(scenario["lambda_rate"] * 10) + int(scenario["p_high"] * 100) + scenario["K"]
    rng = random.Random(seed)
    event_heap = []
    event_counter = 0
    packet_sequence = 0
    event_counter = schedule_event(event_heap, exponential_time(scenario["lambda_rate"], rng), event_counter, "arrival")

    queue = []
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
    replaced_residence_times = []
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
                event_counter = start_service(packet, event_time, rng, scenario["mu_rate"], event_heap, event_counter)
                if event_time >= WARMUP_TIME:
                    counts["accepted_total"] += 1
                    if packet.priority == HIGH_PRIORITY:
                        counts["accepted_high"] += 1
                    else:
                        counts["accepted_low"] += 1
            elif len(queue) < scenario["K"]:
                queue.append(packet)
                packet.was_accepted = True
                if event_time >= WARMUP_TIME:
                    counts["accepted_total"] += 1
                    if packet.priority == HIGH_PRIORITY:
                        counts["accepted_high"] += 1
                    else:
                        counts["accepted_low"] += 1
            elif packet.priority != HIGH_PRIORITY:
                register_drop(packet, event_time, DROP_REASON_LOW_FULL, counts)
            else:
                replacement_index = None
                for index, queued_packet in enumerate(queue):
                    if queued_packet.priority != HIGH_PRIORITY:
                        replacement_index = index
                        break

                if replacement_index is None:
                    register_drop(packet, event_time, DROP_REASON_NO_LOW, counts)
                else:
                    replaced_packet = queue[replacement_index]
                    if current_packet is not None and replaced_packet.id == current_packet.id:
                        raise RuntimeError("The packet in service cannot be replaced.")
                    register_drop(replaced_packet, event_time, DROP_REASON_REPLACED, counts)
                    replaced_packet.was_replaced = True
                    if replaced_packet.arrival_time >= WARMUP_TIME:
                        replaced_residence_times.append(event_time - replaced_packet.arrival_time)
                    queue[replacement_index] = packet
                    packet.was_accepted = True
                    update_replaced_low_count(counts, replaced_packet.arrival_time)
                    if event_time >= WARMUP_TIME:
                        counts["accepted_total"] += 1
                        counts["accepted_high"] += 1

            if len(queue) > scenario["K"]:
                raise RuntimeError("Queue length exceeded K in the intelligent buffer simulation.")

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
                current_packet = queue.pop(0)
                event_counter = start_service(current_packet, event_time, rng, scenario["mu_rate"], event_heap, event_counter)

    if counts["dropped_total"] != counts["dropped_high"] + counts["dropped_low"]:
        raise RuntimeError("Dropped packet totals are inconsistent.")
    if counts["arrivals_total"] != counts["arrivals_high"] + counts["arrivals_low"]:
        raise RuntimeError("Arrival totals are inconsistent.")
    if counts["served_total"] != counts["served_high"] + counts["served_low"]:
        raise RuntimeError("Served totals are inconsistent.")
    if counts["replaced_low"] > counts["dropped_low"]:
        raise RuntimeError("replaced_low cannot exceed dropped_low.")

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
        replaced_residence_times,
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
    """Write all required result files for the intelligent-buffer queue.

    Args:
        replication_rows (list[dict]): Replication-level simulation outputs.
        occupancy_rows (list[dict]): Occupancy-state outputs.
        sample_rows (list[dict]): Time-series rows for the base scenario.
        scenarios (list[dict]): Scenario definitions written into `config.json`.

    Returns:
        Path: Output directory path.

    Raises:
        OSError: If any output file cannot be written.
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
        "description": "Fila M/M/1 com buffer inteligente, descarte seletivo por prioridade e substituicao do pacote de baixa prioridade mais antigo no buffer.",
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
    """Run every scenario and replication for the intelligent-buffer queue.

    Args:
        None: The script uses the module-level constants and scenario builder.

    Returns:
        None: Results are written to disk and completion is printed.

    Raises:
        RuntimeError: If queue invariants are violated during the simulation.
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
