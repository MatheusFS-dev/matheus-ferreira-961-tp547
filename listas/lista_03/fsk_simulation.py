import matplotlib.pyplot as plt
import numpy as np

from fsk_generator import (
    calculate_samples_per_bit,
    generate_fsk_signal,
    generate_random_bits,
)


def add_awgn(signal, snr_db, seed=None):
    """Add white Gaussian noise to a transmitted waveform.

    Args:
        signal (array-like): One-dimensional waveform samples that will be
            corrupted by noise.
        snr_db (float): Signal-to-noise ratio in decibels. Higher values
            produce a cleaner received waveform, while lower values increase
            the probability of detection errors.
        seed (int or None): Seed used by NumPy's random number generator. When
            set to an integer, the generated noise is reproducible. When set to
            `None`, each call produces a different noise realization.

    Returns:
        numpy.ndarray: Noisy version of the input waveform.

    Raises:
        ValueError: If `signal` is not one-dimensional.
        ValueError: If `signal` is empty.
        ValueError: If the average signal power is zero.

    Examples:
        >>> clean_signal = np.array([1.0, -1.0, 1.0, -1.0])
        >>> noisy_signal = add_awgn(clean_signal, snr_db=10.0, seed=1)
        >>> noisy_signal.shape
        (4,)
    """
    signal_array = np.asarray(signal, dtype=float)

    # The channel model assumes a single waveform vector; any other structure
    # would make the receiver segmentation ambiguous.
    if signal_array.ndim != 1:
        raise ValueError("signal must be a one-dimensional sequence.")

    if signal_array.size == 0:
        raise ValueError("signal must contain at least one sample.")

    signal_power = np.mean(signal_array**2)
    if signal_power == 0:
        raise ValueError("signal power must be greater than zero.")

    # Convert the requested SNR into a linear noise power value so the Gaussian
    # noise variance matches the desired channel condition.
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal_array.size)

    return signal_array + noise


def detect_fsk_bits(
    received_signal,
    bit_duration,
    sample_rate,
    frequency_zero,
    frequency_one,
):
    """Detect binary FSK bits by correlating each segment with both carriers.

    This receiver assumes the same bit timing and carrier phase used by the
    transmitter. Under that assumption, the branch with the highest correlation
    indicates which tone was sent during the bit interval.

    Args:
        received_signal (array-like): One-dimensional noisy waveform produced
            by the channel.
        bit_duration (float): Bit interval in seconds. The value must match the
            transmitter configuration exactly.
        sample_rate (float): Sampling frequency in samples per second. The
            value must match the transmitter configuration exactly.
        frequency_zero (float): Carrier frequency in hertz associated with bit
            `0`. The value must match the transmitter configuration exactly.
        frequency_one (float): Carrier frequency in hertz associated with bit
            `1`. The value must match the transmitter configuration exactly.

    Returns:
        numpy.ndarray: Detected bit sequence containing `0` and `1`.

    Raises:
        ValueError: If `received_signal` is not one-dimensional.
        ValueError: If `received_signal` is empty.
        ValueError: If the waveform length is not an integer number of bits.

    Examples:
        >>> bits = np.array([0, 1, 0])
        >>> _, clean_signal, _ = generate_fsk_signal(bits, 0.01, 20000, 1000, 2000)
        >>> detect_fsk_bits(clean_signal, 0.01, 20000, 1000, 2000)
        array([0, 1, 0])
    """
    received_array = np.asarray(received_signal, dtype=float)

    if received_array.ndim != 1:
        raise ValueError("received_signal must be a one-dimensional sequence.")

    if received_array.size == 0:
        raise ValueError("received_signal must contain at least one sample.")

    samples_per_bit = calculate_samples_per_bit(bit_duration, sample_rate)

    # Force exact alignment between the waveform length and the bit boundaries
    # instead of truncating leftover samples.
    if received_array.size % samples_per_bit != 0:
        raise ValueError(
            "received_signal length must be an integer multiple of samples_per_bit."
        )

    number_of_bits = received_array.size // samples_per_bit
    local_time_axis = np.arange(samples_per_bit) / sample_rate
    reference_zero = np.cos(2 * np.pi * frequency_zero * local_time_axis)
    reference_one = np.cos(2 * np.pi * frequency_one * local_time_axis)
    detected_bits = np.zeros(number_of_bits, dtype=int)

    for bit_index in range(number_of_bits):
        start_index = bit_index * samples_per_bit
        end_index = start_index + samples_per_bit
        current_segment = received_array[start_index:end_index]

        # Compare how strongly the received samples match each known carrier.
        correlation_zero = np.sum(current_segment * reference_zero)
        correlation_one = np.sum(current_segment * reference_one)
        detected_bits[bit_index] = 1 if correlation_one > correlation_zero else 0

    return detected_bits


def calculate_bit_error_rate(transmitted_bits, detected_bits):
    """Calculate the bit error rate between the sent and detected sequences.

    Args:
        transmitted_bits (array-like): Original one-dimensional sequence
            containing only `0` and `1`.
        detected_bits (array-like): Receiver output sequence containing only
            `0` and `1`. The shape must match `transmitted_bits`.

    Returns:
        float: Fraction of bits detected incorrectly. A value of `0.0` means
            perfect detection, while `1.0` means every bit was wrong.

    Raises:
        ValueError: If either input is not one-dimensional.
        ValueError: If the sequences do not have the same length.

    Examples:
        >>> calculate_bit_error_rate([0, 1, 1, 0], [0, 0, 1, 0])
        0.25
    """
    transmitted_array = np.asarray(transmitted_bits, dtype=int)
    detected_array = np.asarray(detected_bits, dtype=int)

    if transmitted_array.ndim != 1:
        raise ValueError("transmitted_bits must be a one-dimensional sequence.")

    if detected_array.ndim != 1:
        raise ValueError("detected_bits must be a one-dimensional sequence.")

    if transmitted_array.size != detected_array.size:
        raise ValueError("transmitted_bits and detected_bits must have the same size.")

    return np.mean(transmitted_array != detected_array)


def plot_simulation_results(
    time_axis,
    transmitted_bits,
    transmitted_signal,
    received_signal,
    detected_bits,
    samples_per_bit,
    case_name,
):
    """Plot the main signals involved in the binary FSK simulation.

    Args:
        time_axis (array-like): Time instants in seconds for every waveform
            sample.
        transmitted_bits (array-like): Original bit sequence used by the
            transmitter.
        transmitted_signal (array-like): Clean binary FSK waveform generated by
            the transmitter.
        received_signal (array-like): Noisy waveform after the AWGN channel.
        detected_bits (array-like): Bit sequence produced by the receiver.
        samples_per_bit (int): Number of samples that represent one bit. The
            value is used to expand the bit sequences into stair-step traces.
        case_name (str): Descriptive label for the current simulation case.
            This text is used in the plot titles so the reader can distinguish
            the good, moderate, and very difficult channel conditions.

    Returns:
        None: This function only creates and displays a matplotlib figure.

    Raises:
        ValueError: If the waveform arrays do not share the same length.
        ValueError: If the transmitted and detected bit sequences do not share
            the same length.
        ValueError: If `samples_per_bit` is not greater than zero.

    Examples:
        >>> bits = np.array([0, 1, 0])
        >>> time_axis, clean_signal, samples_per_bit = generate_fsk_signal(
        ...     bits, 0.01, 20000, 1000, 2000
        ... )
        >>> plot_simulation_results(
        ...     time_axis,
        ...     bits,
        ...     clean_signal,
        ...     clean_signal,
        ...     bits,
        ...     samples_per_bit,
        ...     "Example case",
        ... )
    """
    time_array = np.asarray(time_axis, dtype=float)
    transmitted_bits_array = np.asarray(transmitted_bits, dtype=int)
    transmitted_signal_array = np.asarray(transmitted_signal, dtype=float)
    received_signal_array = np.asarray(received_signal, dtype=float)
    detected_bits_array = np.asarray(detected_bits, dtype=int)

    if samples_per_bit <= 0:
        raise ValueError("samples_per_bit must be greater than zero.")

    if time_array.size != transmitted_signal_array.size:
        raise ValueError("time_axis and transmitted_signal must have the same size.")

    if time_array.size != received_signal_array.size:
        raise ValueError("time_axis and received_signal must have the same size.")

    if transmitted_bits_array.size != detected_bits_array.size:
        raise ValueError("transmitted_bits and detected_bits must have the same size.")

    transmitted_bits_step = np.repeat(transmitted_bits_array, samples_per_bit)
    detected_bits_step = np.repeat(detected_bits_array, samples_per_bit)

    # Show the bit pattern first so the reader can associate every waveform
    # segment with its corresponding symbol.
    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.step(time_array, transmitted_bits_step, where="post", color="tab:blue")
    plt.ylim(-0.2, 1.2)
    plt.ylabel("Bit value")
    plt.title(f"{case_name}: Transmitted bit sequence")
    plt.grid(True, alpha=0.3)

    # Plot the clean waveform to highlight how the transmitted frequency
    # changes when the bit value switches between 0 and 1.
    plt.subplot(4, 1, 2)
    plt.plot(time_array, transmitted_signal_array, color="tab:green", linewidth=1.2)
    plt.ylabel("Amplitude")
    plt.title(f"{case_name}: Generated FSK signal")
    plt.grid(True, alpha=0.3)

    # Plot the noisy waveform so the effect of the channel can be compared
    # directly against the clean transmitted signal.
    plt.subplot(4, 1, 3)
    plt.plot(time_array, received_signal_array, color="tab:red", linewidth=1.0)
    plt.ylabel("Amplitude")
    plt.title(f"{case_name}: Received signal with AWGN")
    plt.grid(True, alpha=0.3)

    # Display the detected sequence last to make the transmission/reception
    # comparison straightforward.
    plt.subplot(4, 1, 4)
    plt.step(time_array, detected_bits_step, where="post", color="tab:purple")
    plt.ylim(-0.2, 1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Bit value")
    plt.title(f"{case_name}: Detected bit sequence")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_simulation(case_name, snr_db, seed):
    """Execute one complete binary FSK transmission and reception case.

    The function generates a random bit stream, modulates it using binary FSK,
    adds AWGN noise, detects the received bits, prints the numerical results,
    and displays the main plots for one predefined channel condition.

    Args:
        case_name (str): Descriptive label printed in the console output and
            shown in the plot titles. This argument does not change the signal
            generation itself; it only identifies the channel condition being
            simulated.
        snr_db (float): Signal-to-noise ratio in decibels for the AWGN channel.
            Larger values create an easier detection case, while smaller values
            create harder cases with more decision errors.
        seed (int): Seed used to generate the transmitted bits and the channel
            noise. Reusing the same seed across different cases keeps the bit
            pattern fixed so only the channel difficulty changes.

    Returns:
        dict: Dictionary containing the transmitted bits, detected bits, BER,
            the main waveform arrays used in the simulation, the case label,
            and the SNR used by the channel.

    Raises:
        ValueError: Propagates parameter validation errors raised by the helper
            functions if the simulation constants are changed to invalid values
            or if an invalid SNR-dependent waveform is produced.

    Examples:
        >>> results = run_simulation("Case 1 - Good channel", 10.0, seed=7)
        >>> results["case_name"]
        'Case 1 - Good channel'
        >>> "bit_error_rate" in results
        True
    """
    # Keep the simulation parameters explicit so the relationship between bit
    # duration, sampling, and carrier spacing is easy to inspect.
    number_of_bits = 10
    bit_duration = 0.01
    sample_rate = 20000
    frequency_zero = 1000
    frequency_one = 2000
    amplitude = 1.0

    transmitted_bits = generate_random_bits(number_of_bits, seed=seed)
    time_axis, transmitted_signal, samples_per_bit = generate_fsk_signal(
        transmitted_bits,
        bit_duration,
        sample_rate,
        frequency_zero,
        frequency_one,
        amplitude=amplitude,
    )
    received_signal = add_awgn(transmitted_signal, snr_db=snr_db, seed=seed + 1)
    detected_bits = detect_fsk_bits(
        received_signal,
        bit_duration,
        sample_rate,
        frequency_zero,
        frequency_one,
    )
    bit_error_rate = calculate_bit_error_rate(transmitted_bits, detected_bits)

    print(case_name)
    print("Binary FSK simulation")
    print(f"Number of bits: {number_of_bits}")
    print(f"Bit duration: {bit_duration:.4f} s")
    print(f"Sample rate: {sample_rate} samples/s")
    print(f"Frequency for bit 0: {frequency_zero} Hz")
    print(f"Frequency for bit 1: {frequency_one} Hz")
    print(f"SNR: {snr_db:.2f} dB")
    print("Transmitted bits:", "".join(str(bit) for bit in transmitted_bits))
    print("Detected bits:   ", "".join(str(bit) for bit in detected_bits))
    print(f"Bit error rate: {bit_error_rate:.4f}")

    plot_simulation_results(
        time_axis,
        transmitted_bits,
        transmitted_signal,
        received_signal,
        detected_bits,
        samples_per_bit,
        case_name,
    )

    return {
        "case_name": case_name,
        "snr_db": snr_db,
        "time_axis": time_axis,
        "transmitted_bits": transmitted_bits,
        "transmitted_signal": transmitted_signal,
        "received_signal": received_signal,
        "detected_bits": detected_bits,
        "bit_error_rate": bit_error_rate,
        "samples_per_bit": samples_per_bit,
    }


def run_simulation_case_1():
    """Run the easiest FSK case with a high-SNR channel.

    Args:
        None: This wrapper selects the fixed case label and SNR associated with
            the easiest simulation condition requested by the assignment.

    Returns:
        dict: Simulation results returned by `run_simulation` for Case 1.

    Raises:
        ValueError: Propagates any validation error raised by
            `run_simulation`.

    Examples:
        >>> results = run_simulation_case_1()
        >>> results["case_name"]
        'Case 1 - Good channel'
    """
    return run_simulation("Case 1 - Good channel", 10.0, seed=7)


def run_simulation_case_2():
    """Run the moderate FSK case with a lower-SNR channel.

    Args:
        None: This wrapper selects the fixed case label and SNR associated with
            the intermediate simulation difficulty requested by the assignment.

    Returns:
        dict: Simulation results returned by `run_simulation` for Case 2.

    Raises:
        ValueError: Propagates any validation error raised by
            `run_simulation`.

    Examples:
        >>> results = run_simulation_case_2()
        >>> results["case_name"]
        'Case 2 - Moderate channel'
    """
    return run_simulation("Case 2 - Moderate channel", -19.0, seed=7)


def run_simulation_case_3():
    """Run the hardest FSK case with a very low-SNR channel.

    Args:
        None: This wrapper selects the fixed case label and SNR associated with
            the most difficult simulation condition requested by the
            assignment.

    Returns:
        dict: Simulation results returned by `run_simulation` for Case 3.

    Raises:
        ValueError: Propagates any validation error raised by
            `run_simulation`.

    Examples:
        >>> results = run_simulation_case_3()
        >>> results["case_name"]
        'Case 3 - Very difficult channel'
    """
    return run_simulation("Case 3 - Very difficult channel", -30.0, seed=7)


def run_all_simulations():
    """Run the three predefined FSK channel cases in sequence.

    Args:
        None: This function always runs the same three cases so the output
            order remains stable: good, moderate, and very difficult.

    Returns:
        list[dict]: List containing the result dictionary returned by each case
            in execution order.

    Raises:
        ValueError: Propagates any validation error raised by the individual
            case functions.

    Examples:
        >>> results = run_all_simulations()
        >>> len(results)
        3
    """
    simulation_results = []
    simulation_functions = [
        run_simulation_case_1,
        run_simulation_case_2,
        run_simulation_case_3,
    ]

    for case_index, simulation_function in enumerate(simulation_functions):
        # Separate the console blocks so the three reports are easier to read.
        if case_index > 0:
            print()

        simulation_results.append(simulation_function())

    return simulation_results


if __name__ == "__main__":
    run_all_simulations()
