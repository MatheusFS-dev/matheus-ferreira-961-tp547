import numpy as np


def calculate_samples_per_bit(bit_duration, sample_rate):
    """Calculate the exact number of samples used to represent one bit.

    This helper keeps the simulation explicit: each bit must occupy an
    integer number of samples, otherwise the generated waveform would not
    align cleanly with the receiver segmentation step.

    Args:
        bit_duration (float): Bit interval in seconds. The value must be
            strictly positive and chosen so that `bit_duration * sample_rate`
            is an integer.
        sample_rate (float): Sampling frequency in samples per second. The
            value must be strictly positive.

    Returns:
        int: Number of samples that represent one bit.

    Raises:
        ValueError: If `bit_duration` is not positive.
        ValueError: If `sample_rate` is not positive.
        ValueError: If one bit does not map to an integer number of samples.

    Examples:
        >>> calculate_samples_per_bit(0.01, 20000)
        200
    """
    # Reject invalid timing parameters instead of silently rounding them.
    if bit_duration <= 0:
        raise ValueError("bit_duration must be positive.")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    raw_samples_per_bit = bit_duration * sample_rate
    rounded_samples_per_bit = int(round(raw_samples_per_bit))

    # Force the user to choose compatible parameters for a clean simulation.
    if not np.isclose(raw_samples_per_bit, rounded_samples_per_bit):
        raise ValueError(
            "bit_duration * sample_rate must be an integer number of samples."
        )

    return rounded_samples_per_bit


def generate_random_bits(number_of_bits, seed=None):
    """Generate a binary sequence for the FSK transmitter.

    Args:
        number_of_bits (int): Total number of bits to generate. The value must
            be greater than zero.
        seed (int or None): Seed used by NumPy's random number generator. When
            set to an integer, the generated sequence is reproducible. When set
            to `None`, the sequence changes from run to run.

    Returns:
        numpy.ndarray: One-dimensional array containing only `0` and `1`.

    Raises:
        ValueError: If `number_of_bits` is not greater than zero.

    Examples:
        >>> generate_random_bits(5, seed=7)
        array([1, 1, 1, 1, 1])
    """
    # The sequence length must be explicit
    if number_of_bits <= 0:
        raise ValueError("number_of_bits must be greater than zero.")

    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=number_of_bits, dtype=int)


def generate_fsk_signal(
    bits,
    bit_duration,
    sample_rate,
    frequency_zero,
    frequency_one,
    amplitude=1.0,
):
    """Generate a binary FSK waveform with one carrier per bit value.

    This implementation uses the simplest educational model: the cosine phase
    restarts at the beginning of every bit interval. Bit `0` uses
    `frequency_zero`, and bit `1` uses `frequency_one`.

    Args:
        bits (array-like): One-dimensional sequence containing only `0` and
            `1`. Each element defines which carrier frequency will be used
            during one bit interval.
        bit_duration (float): Bit interval in seconds. The value must be
            strictly positive and compatible with `sample_rate`.
        sample_rate (float): Sampling frequency in samples per second. The
            value must be strictly positive.
        frequency_zero (float): Carrier frequency in hertz used when the
            transmitted bit is `0`. The value must be strictly positive.
        frequency_one (float): Carrier frequency in hertz used when the
            transmitted bit is `1`. The value must be strictly positive.
        amplitude (float): Peak amplitude of the transmitted cosine waveform.
            The value must be strictly positive.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, int]: Tuple containing:
            - `time_axis`: time instants for every generated sample in seconds.
            - `signal`: binary FSK waveform samples.
            - `samples_per_bit`: number of samples assigned to each bit.

    Raises:
        ValueError: If `bits` is not one-dimensional.
        ValueError: If `bits` contains values other than `0` and `1`.
        ValueError: If any timing, frequency, or amplitude parameter is
            invalid.

    Examples:
        >>> bits = np.array([0, 1, 0])
        >>> _, signal, samples_per_bit = generate_fsk_signal(
        ...     bits, 0.01, 20000, 1000, 2000
        ... )
        >>> signal.size == bits.size * samples_per_bit
        True
    """
    bit_array = np.asarray(bits, dtype=int)

    # The receiver depends on a single linear list of bits, so multidimensional
    # inputs are rejected instead of flattened implicitly.
    if bit_array.ndim != 1:
        raise ValueError("bits must be a one-dimensional sequence.")

    if bit_array.size == 0:
        raise ValueError("bits must contain at least one element.")

    if not np.all((bit_array == 0) | (bit_array == 1)):
        raise ValueError("bits must contain only 0 and 1.")

    if frequency_zero <= 0:
        raise ValueError("frequency_zero must be positive.")

    if frequency_one <= 0:
        raise ValueError("frequency_one must be positive.")

    if amplitude <= 0:
        raise ValueError("amplitude must be positive.")

    samples_per_bit = calculate_samples_per_bit(bit_duration, sample_rate)
    total_samples = bit_array.size * samples_per_bit

    # Build a global time axis so plots and post-processing can use the same
    # sample positions without recomputing them.
    time_axis = np.arange(total_samples) / sample_rate
    signal = np.zeros(total_samples, dtype=float)

    # Reuse the local time base inside each bit interval because the phase is
    # intentionally restarted for every transmitted symbol.
    local_time_axis = np.arange(samples_per_bit) / sample_rate

    for bit_index, bit_value in enumerate(bit_array):
        start_index = bit_index * samples_per_bit
        end_index = start_index + samples_per_bit

        # Select the carrier frequency that corresponds to the current bit.
        carrier_frequency = frequency_one if bit_value == 1 else frequency_zero
        signal[start_index:end_index] = amplitude * np.cos(
            2 * np.pi * carrier_frequency * local_time_axis
        )

    return time_axis, signal, samples_per_bit
