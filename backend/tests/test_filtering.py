"""Filtering tests: energy outside the retained band must really vanish."""

import numpy as np
import pytest

from app.dft import dft
from app.errors import BadRequest
from app.filtering import apply_filter


def _two_tone(n=256, fs=256.0):
    # 10 Hz tone + 80 Hz tone, both on-bin.
    t = np.arange(n) / fs
    return np.cos(2 * np.pi * 10 * t) + 0.75 * np.cos(2 * np.pi * 80 * t)


def test_lowpass_kills_high_frequency_energy():
    n, fs = 256, 256.0
    x = _two_tone(n, fs)
    result = apply_filter(x, fs, "lowpass", cutoff_high=20.0)
    y, xk = result["filtered"], result["spectrum"]

    # Nothing above 20 Hz survives in the masked spectrum.
    freqs = np.arange(n) * fs / n
    folded = np.minimum(freqs, fs - freqs)
    assert np.allclose(xk[folded > 20.0], 0.0, atol=1e-10)
    assert result["residual_high_energy"] < 1e-20

    # In the time domain the 80 Hz component is gone: compare against a pure
    # 10 Hz tone (Gibbs is absent here since the cut falls cleanly between
    # on-bin components).
    t = np.arange(n) / fs
    expected = np.cos(2 * np.pi * 10 * t)
    np.testing.assert_allclose(y, expected, atol=1e-9)
    assert result["output_energy"] == pytest.approx(n / 2, rel=1e-9)


def test_highpass_kills_low_frequency_energy():
    n, fs = 256, 256.0
    x = _two_tone(n, fs)
    result = apply_filter(x, fs, "highpass", cutoff_low=20.0)
    y, xk = result["filtered"], result["spectrum"]
    freqs = np.arange(n) * fs / n
    folded = np.minimum(freqs, fs - freqs)
    assert np.allclose(xk[folded < 20.0], 0.0, atol=1e-10)
    t = np.arange(n) / fs
    np.testing.assert_allclose(y, 0.75 * np.cos(2 * np.pi * 80 * t), atol=1e-9)


def test_bandpass_keeps_only_middle_band():
    n, fs = 256, 256.0
    t = np.arange(n) / fs
    x = (
        np.cos(2 * np.pi * 5 * t)
        + np.cos(2 * np.pi * 40 * t)
        + np.cos(2 * np.pi * 100 * t)
    )
    result = apply_filter(x, fs, "bandpass", cutoff_low=20.0, cutoff_high=60.0)
    xk = result["spectrum"]
    assert abs(xk[5]) < 1e-10 and abs(xk[n - 5]) < 1e-10
    assert abs(xk[100]) < 1e-10 and abs(xk[n - 100]) < 1e-10
    assert abs(xk[40]) > 100.0 and abs(xk[n - 40]) > 100.0
    np.testing.assert_allclose(
        result["filtered"], np.cos(2 * np.pi * 40 * t), atol=1e-9
    )


def test_filter_returns_real_signal_and_keeps_energy_when_all_pass():
    n, fs = 256, 256.0
    x = _two_tone(n, fs)
    result = apply_filter(x, fs, "lowpass", cutoff_high=fs / 2)
    assert np.isrealobj(result["filtered"])
    np.testing.assert_allclose(result["filtered"], x, atol=1e-10)
    assert result["output_energy"] == pytest.approx(result["input_energy"], rel=1e-9)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mode": "bandstop"}, "filter mode"),
        ({"mode": "lowpass", "cutoff_high": 200.0}, "Nyquist"),
        ({"mode": "lowpass", "cutoff_high": -1.0}, "Nyquist"),
        ({"mode": "highpass", "cutoff_low": -1.0}, "Nyquist"),
        (
            {"mode": "bandpass", "cutoff_low": 60.0, "cutoff_high": 40.0},
            "strictly greater",
        ),
        (
            {"mode": "bandpass", "cutoff_low": 40.0, "cutoff_high": 40.0},
            "strictly greater",
        ),
        ({"mode": "bandpass", "cutoff_low": 10.0}, "cutoff_high"),
    ],
)
def test_invalid_bands_rejected(kwargs, match):
    with pytest.raises(BadRequest, match=match):
        apply_filter(np.zeros(64), 256.0, **kwargs)


def test_nonpositive_fs_rejected():
    with pytest.raises(BadRequest, match="sampling rate"):
        apply_filter(np.zeros(64), 0.0, "lowpass", cutoff_high=10.0)
