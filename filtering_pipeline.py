"""Multi-stage noise filtering pipeline for bioacoustic bird detection."""

import numpy as np
from scipy.signal import butter, lfilter, iirnotch, filtfilt


def butter_highpass(cutoff, fs, order=5):
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


def apply_highpass(data, cutoff=300, fs=44100, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data).astype(data.dtype)


def apply_bandpass(data, lowcut=500, highcut=22000, fs=44100, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data).astype(data.dtype)


def apply_notch(data, freq, fs=44100, Q=30):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data).astype(data.dtype)


def apply_notch_filters(data, fs=44100):
    """Apply notch filters for insect chorus (crickets 4-7 kHz, cicadas 6-10 kHz)."""
    for freq in [4500, 5500, 6500, 7500, 8500, 9500]:
        data = apply_notch(data, freq, fs)
    return data


def wiener_filter_multi_band(data, fs=44100, stationary=True):
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
                                   prop_decrease=0.8, stationary=True)
        else:
            # Non-stationary mode: use time-frequency masking (slower)
            reduced = reduce_noise(y=data.astype(np.float64), sr=fs, prop_decrease=0.8)
        return reduced.astype(data.dtype)
    except Exception as e:
        warnings.warn(f"Wiener filter failed ({e}), falling back to spectral gating")
        return spectral_gate(data, fs)


def spectral_gate(data, fs=44100, noise_floor_db=-50):
    """Simple spectral gating fallback using STFT filtering."""
    from scipy.signal import stft, istft
    nperseg = 1024
    f, t, Zxx = stft(data.astype(np.float64), fs, nperseg=nperseg, noverlap=768)
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


def normalize_audio(data, target_max=32767):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return (data / max_val * target_max).astype(np.int16)
    return data.astype(np.int16)


def full_filter_pipeline(data, fs=44100,
                          hp_cutoff=300, lp_cutoff=22000,
                          enable_wiener=True, enable_notch=True,
                          wiener_fallback=True, stationary_wiener=True):
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
