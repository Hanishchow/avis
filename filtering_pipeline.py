"""Multi-stage noise filtering pipeline for bioacoustic bird detection."""

import numpy as np
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from config import SAMPLE_RATE, FILTER_ORDER, NOTCH_FREQS, NOTCH_Q, WIENER_PROP_DECREASE, NOISE_FLOOR_DB, SPECTRAL_GATE_NPERSEG, SPECTRAL_GATE_NOVERLAP, NORMALIZE_TARGET_MAX


def butter_highpass(cutoff, fs, order=None):
    if order is None:
        order = FILTER_ORDER
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band", analog=False)
    return b, a


def apply_highpass(data, cutoff=None, fs=None, order=None):
    if cutoff is None:
        cutoff = 300
    if fs is None:
        fs = SAMPLE_RATE
    if order is None:
        order = FILTER_ORDER
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data).astype(data.dtype)


def apply_bandpass(data, lowcut=500, highcut=None, fs=None, order=None):
    if highcut is None:
        highcut = 22000
    if fs is None:
        fs = SAMPLE_RATE
    if order is None:
        order = FILTER_ORDER
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data).astype(data.dtype)


def apply_notch(data, freq, fs=None, Q=None):
    if fs is None:
        fs = SAMPLE_RATE
    if Q is None:
        Q = NOTCH_Q
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data).astype(data.dtype)


def apply_notch_filters(data, fs=None):
    """Apply notch filters for insect chorus (crickets 4-7 kHz, cicadas 6-10 kHz)."""
    if fs is None:
        fs = SAMPLE_RATE
    for freq in NOTCH_FREQS:
        data = apply_notch(data, freq, fs)
    return data


def wiener_filter_multi_band(data, fs=None, stationary=None):
    if fs is None:
        fs = SAMPLE_RATE
    if stationary is None:
        stationary = True
    """Multi-band adaptive Wiener filter for wind and rain noise reduction.
    Falls back to spectral gating if filter fails to converge.
    Use stationary=True (default) for faster processing (~4x); use stationary=False
    for non-stationary noise (wind/rain) at higher computational cost."""
    import warnings
    try:
        from noisereduce import reduce_noise
        if stationary:
            # Stationary mode: estimate noise from first 0.5s
            noise_len = int(0.5 * fs)
            noise_sample = data[:noise_len] if len(data) > noise_len else data[:len(data)//4]
            reduced = reduce_noise(y=data.astype(np.float64), sr=fs, y_noise=noise_sample.astype(np.float64),
                                   prop_decrease=WIENER_PROP_DECREASE, stationary=True)
        else:
            reduced = reduce_noise(y=data.astype(np.float64), sr=fs, prop_decrease=WIENER_PROP_DECREASE)
        return reduced.astype(data.dtype)
    except Exception as e:
        warnings.warn(f"Wiener filter failed ({e}), falling back to spectral gating")
        return spectral_gate(data, fs)


def spectral_gate(data, fs=None, noise_floor_db=None):
    """Simple spectral gating fallback using STFT filtering."""
    if fs is None:
        fs = SAMPLE_RATE
    if noise_floor_db is None:
        noise_floor_db = NOISE_FLOOR_DB
    from scipy.signal import stft, istft
    f, t, Zxx = stft(data.astype(np.float64), fs, nperseg=SPECTRAL_GATE_NPERSEG, noverlap=SPECTRAL_GATE_NOVERLAP)
    threshold = 10 ** (noise_floor_db / 10)
    magnitude = np.abs(Zxx)
    mask = magnitude > threshold * np.median(magnitude, axis=1, keepdims=True)
    Zxx_clean = Zxx * mask
    _, data_clean = istft(Zxx_clean, fs)
    if len(data_clean) > len(data):
        data_clean = data_clean[:len(data)]
    elif len(data_clean) < len(data):
        data_clean = np.pad(data_clean, (0, len(data) - len(data_clean)))
    return data_clean.astype(data.dtype)


def normalize_audio(data, target_max=None):
    if target_max is None:
        target_max = NORMALIZE_TARGET_MAX
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return (data / max_val * target_max).astype(np.int16)
    return data.astype(np.int16)


def full_filter_pipeline(data, fs=None,
                          hp_cutoff=None, lp_cutoff=None,
                          enable_wiener=None, enable_notch=None,
                          wiener_fallback=True, stationary_wiener=None):
    if fs is None:
        fs = SAMPLE_RATE
    if hp_cutoff is None:
        hp_cutoff = 300
    if lp_cutoff is None:
        lp_cutoff = 22000
    if enable_wiener is None:
        enable_wiener = True
    if enable_notch is None:
        enable_notch = True
    if stationary_wiener is None:
        stationary_wiener = True
    """Chain all filtering stages in sequence."""
    signal = data.astype(np.float64)

    # 1. High-pass filter (remove low-frequency rumble)
    signal = apply_highpass(signal, cutoff=hp_cutoff, fs=fs)

    # 2. Bandpass filter (focus on bird frequency range)
    signal = apply_bandpass(signal, lowcut=500, highcut=lp_cutoff, fs=fs)

    # 3. Multi-band adaptive Wiener filter
    if enable_wiener:
        signal = wiener_filter_multi_band(signal, fs, stationary=stationary_wiener)

    # 4. Spectral reduction (via noise reduce, integrated in wiener step)

    # 5. Notch filters for insect chorus
    if enable_notch:
        signal = apply_notch_filters(signal, fs)

    # 6. Amplitude normalization
    signal = normalize_audio(signal)

    return signal
