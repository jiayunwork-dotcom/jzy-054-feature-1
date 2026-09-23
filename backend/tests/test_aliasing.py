"""Aliasing criterion and sampling-demo tests.

The Nyquist decision must hold by rule — not by eyeballing a plot.
"""

import numpy as np
import pytest

from app.aliasing import apparent_frequency, is_aliased, sampling_demo
from app.errors import BadRequest


@pytest.mark.parametrize(
    "f,fs,expected_alias,expected_apparent",
    [
        (50, 200, False, 50),
        (100, 200, False, 100),     # exactly Nyquist: not aliased
        (150, 200, True, 50),       # folds back to 50
        (300, 200, True, 100),
        (250, 200, True, 50),
        (75, 100, True, 25),
        (0, 100, False, 0),
        (12.5, 25, False, 12.5),
        (17.3, 25, True, 7.7),
    ],
)
def test_alias_rule(f, fs, expected_alias, expected_apparent):
    assert is_aliased(f, fs) is expected_alias
    assert apparent_frequency(f, fs) == pytest.approx(
        expected_apparent, abs=1e-10
    )


def test_apparent_frequency_always_in_nyquist_band():
    for f in np.linspace(0, 1000, 201):
        for fs in (10, 50, 100, 333):
            fa = apparent_frequency(f, fs)
            assert 0.0 <= fa <= fs / 2.0 + 1e-9


def test_demo_samples_equal_true_tone_at_sample_instants():
    out = sampling_demo(37.0, 100.0, n_samples=20)
    # Sample values are exact evaluations of the true signal.
    t = np.arange(20) / 100.0
    np.testing.assert_allclose(out["sample_y"], np.cos(2 * np.pi * 37 * t))


def test_demo_reconstruction_follows_apparent_tone_when_aliased():
    fs = 100.0
    f_true, f_app = 70.0, 30.0
    out = sampling_demo(f_true, fs, n_samples=48)
    assert out["aliased"] is True
    assert out["apparent_freq_hz"] == pytest.approx(f_app)

    # Away from the finite-window edges, sinc interpolation of the samples
    # follows the *apparent* 30 Hz tone, not the true 70 Hz tone.
    t = np.asarray(out["reconstructed_t"])
    y = np.asarray(out["reconstructed_y"])
    mid = (t > 0.15) & (t < 0.33)
    expected_apparent = np.cos(2 * np.pi * f_app * t[mid])
    expected_true = np.cos(2 * np.pi * f_true * t[mid])
    np.testing.assert_allclose(y[mid], expected_apparent, atol=0.05)
    assert np.max(np.abs(y[mid] - expected_true)) > 0.5


def test_demo_reconstruction_recovers_tone_below_nyquist():
    out = sampling_demo(20.0, 100.0, n_samples=48)
    assert out["aliased"] is False
    t = np.asarray(out["reconstructed_t"])
    y = np.asarray(out["reconstructed_y"])
    mid = (t > 0.15) & (t < 0.33)
    np.testing.assert_allclose(y[mid], np.cos(2 * np.pi * 20 * t[mid]), atol=0.05)


def test_demo_validates_inputs():
    with pytest.raises(BadRequest, match="sampling rate"):
        sampling_demo(10.0, 0.0)
    with pytest.raises(BadRequest, match="non-negative"):
        sampling_demo(-1.0, 100.0)
    with pytest.raises(BadRequest, match="at least 4"):
        sampling_demo(10.0, 100.0, n_samples=2)
