"""Unit tests for the filtering pipeline."""

import numpy as np
from filtering_pipeline import (
    apply_highpass,
    apply_bandpass,
    apply_notch_filters,
    wiener_filter_multi_band,
    normalize_audio,
    full_filter_pipeline,
)


def test_highpass_shapes():
    fs = 44100
    data = np.random.randn(fs * 3).astype(np.float64)
    out = apply_highpass(data, cutoff=300, fs=fs)
    assert data.shape == out.shape, f"Shape mismatch: {data.shape} vs {out.shape}"
    print("PASS: test_highpass_shapes")


def test_highpass_removes_low_freq():
    fs = 44100
    t = np.linspace(0, 1, fs)
    # 50 Hz signal (should be attenuated by 300 Hz high-pass)
    low_freq = np.sin(2 * np.pi * 50 * t)
    # 2 kHz signal (should be preserved)
    high_freq = np.sin(2 * np.pi * 2000 * t)
    mixed = (low_freq + high_freq) * 0.5
    out = apply_highpass(mixed.astype(np.float64), cutoff=300, fs=fs)
    # Power of 50 Hz component should be reduced
    power_before = np.mean(low_freq**2)
    power_after = np.mean(out**2)
    ratio = power_after / power_before if power_before > 0 else 1
    assert ratio < 0.5, f"Low freq not attenuated: ratio={ratio:.3f}"
    print(f"PASS: test_highpass_removes_low_freq (ratio={ratio:.3f})")


def test_bandpass_preserves_mid_range():
    fs = 44100
    t = np.linspace(0, 1, fs)
    signal = np.sin(2 * np.pi * 2000 * t)  # 2 kHz (inside passband 500-22000)
    out = apply_bandpass(signal.astype(np.float64), lowcut=500, highcut=22000, fs=fs)
    power_in = np.mean(signal**2)
    power_out = np.mean(out.astype(np.float64)**2)
    assert power_out > 0.5 * power_in, f"Mid freq attenuated: {power_out/power_in:.3f}"
    print(f"PASS: test_bandpass_preserves_mid_range")


def test_notch_filters():
    fs = 44100
    t = np.linspace(0, 1, fs)
    # Pure 5 kHz tone (cricket range)
    signal = np.sin(2 * np.pi * 5500 * t)
    out = apply_notch_filters(signal.astype(np.float64), fs=fs)
    power_in = np.mean(signal**2)
    power_out = np.mean(out.astype(np.float64)**2)
    assert power_out < 0.5 * power_in, f"Notch did not attenuate: {power_out/power_in:.3f}"
    print(f"PASS: test_notch_filters")


def test_normalize_audio():
    data = np.array([1000, -2000, 3000, -4000], dtype=np.float64)
    normed = normalize_audio(data)
    assert np.max(np.abs(normed)) == 32767
    assert normed.dtype == np.int16
    print("PASS: test_normalize_audio")


def test_full_pipeline():
    fs = 44100
    data = np.random.randn(fs * 3).astype(np.float64)
    out = full_filter_pipeline(data, fs=fs)
    assert len(out) == len(data), f"Length mismatch: {len(out)} vs {len(data)}"
    assert out.dtype == np.int16
    print("PASS: test_full_pipeline")


if __name__ == "__main__":
    test_highpass_shapes()
    test_highpass_removes_low_freq()
    test_bandpass_preserves_mid_range()
    test_notch_filters()
    test_normalize_audio()
    test_full_pipeline()
    print("\nAll filtering tests passed!")
