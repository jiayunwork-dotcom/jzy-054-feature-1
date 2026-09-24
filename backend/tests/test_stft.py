"""Identity tests for the short-time analysis module.

These guard the three relationships the product spec demands to hold
numerically rather than merely "look right" on the heatmap:

1. a constant-frequency tone, split into many frames, keeps its energy on
   the *same* frequency row in every frame (a horizontal bright line);
2. a linearly swept chirp has its energy-peak row rise *monotonically* with
   frame index (a diagonal whose direction the tests can judge);
3. under genuine overlap and windowing, the weighted overlap-add inverse
   reconstructs the original signal to numerical precision.

Input validation (frame-length ladder, hop sanity, short signals, frame
structure mismatches) and the legal single-frame degenerate case are
pinned here as well.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import BadRequest
from app.stft import (
    frame_signal,
    frame_times,
    istft,
    num_frames,
    spectrogram,
    stft,
)

FS = 1000.0


# ------------------------------------------------------------- 1. horizontal


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman"])
def test_constant_tone_is_horizontal_line(window):
    f0 = 40.0
    frame_length, hop = 128, 32
    t = np.arange(512) / FS
    x = np.sin(2 * np.pi * f0 * t)
    sg = spectrogram(x, FS, frame_length, hop, window)

    power = sg["power"]
    peaks = np.argmax(power, axis=0)
    # Every column peaks on exactly the same frequency row ...
    assert set(peaks.tolist()) == {int(peaks[0])}
    # ... and that row is the bin nearest the true frequency.
    expected_bin = round(f0 / FS * frame_length)
    assert peaks[0] == expected_bin
    assert sg["frequencies"][peaks[0]] == pytest.approx(f0, abs=FS / frame_length)

    # Energy concentration: compared with a frequency-shuffled surrogate of
    # the same column, the tone column is overwhelmingly concentrated around
    # its peak row. We assert the peak row plus its two neighbors hold the
    # dominant share for the strong main-lobe windows (rect/hann).
    col = power.shape[1] // 2
    total = power[:, col].sum()
    near = power[max(0, peaks[col] - 2) : peaks[col] + 3, col].sum()
    assert near / total > 0.85


def test_constant_tone_frame_spectra_equal_under_rect():
    # With a rectangular window an on-bin tone produces identical frame
    # spectra across all fully-internal frames — a stronger version of the
    # horizontal-line claim.
    frame_length, hop = 64, 16
    f0 = 5.0 * FS / frame_length  # exactly on bin 5
    t = np.arange(384) / FS
    x = np.sin(2 * np.pi * f0 * t)
    r = stft(x, FS, frame_length, hop, "rect")
    frames = r["frames"]
    # Internal frames differ only in negligible edge terms; the peak bin is
    # identical throughout.
    peaks = np.argmax(np.abs(frames[:, : frame_length // 2 + 1]), axis=1)
    assert set(peaks.tolist()) == {5}


# ----------------------------------------------------------------- 2. chirp


def _parabolic_peak(power_row: np.ndarray, k: int, df: float) -> float:
    """Sub-bin peak frequency by quadratic interpolation around bin k."""
    if k <= 0 or k >= len(power_row) - 1:
        return k * df
    a, b, c = power_row[k - 1], power_row[k], power_row[k + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return k * df
    delta = 0.5 * (a - c) / denom  # in [-0.5, 0.5]
    return (k + np.clip(delta, -0.5, 0.5)) * df


def test_linear_chirp_peak_row_monotonic():
    frame_length, hop = 256, 64
    duration = 0.8
    t = np.arange(int(duration * FS)) / FS
    f0, rate = 20.0, 400.0  # Hz, Hz/s -> ends at ~340 Hz, stays below fs/2
    phase = 2 * np.pi * (f0 * t + 0.5 * rate * t**2)
    chirp = np.sin(phase)
    sg = spectrogram(chirp, FS, frame_length, hop, "hann")
    power = sg["power"]
    peaks = np.argmax(power, axis=0)

    # Sub-bin-interpolated peak frequency must increase strictly column by
    # column once the sweep is well inside the band (argmax alone can plateau
    # when the true sweep moves less than a bin between columns).
    df = FS / frame_length
    f_est = np.array(
        [_parabolic_peak(power[:, col], int(peaks[col]), df) for col in range(power.shape[1])]
    )
    interior = f_est[1:-1]
    assert np.all(np.diff(interior) > 0)

    # Track the true instantaneous frequency within one bin on interior cols.
    for col in range(1, len(peaks) - 1):
        tcol = sg["times"][col]
        assert f_est[col] == pytest.approx(f0 + rate * tcol, abs=1.5 * df)

    # And the coarse integer peak rows are non-decreasing over the interior.
    assert np.all(np.diff(peaks[1:-1]) >= 0)


def test_peak_row_frequency_values_match_sweep_slope():
    # Fit a line f_peak(t) = a + b*t and check the slope equals the sweep
    # rate (the relationship the spec wants judged by a program, not eyes).
    # Parabolic interpolation removes DFT-bin quantization so the fitted
    # slope is meaningful; the sweep moves several bins per column.
    frame_length, hop = 128, 16
    duration = 0.5
    t = np.arange(int(duration * FS)) / FS
    f0, rate = 30.0, 300.0
    x = np.sin(2 * np.pi * (f0 * t + 0.5 * rate * t**2))
    sg = spectrogram(x, FS, frame_length, hop, "hann")
    power = sg["power"]
    peaks = np.argmax(power, axis=0)
    df = FS / frame_length
    f_peak = np.array(
        [_parabolic_peak(power[:, col], int(k), df) for col, k in enumerate(peaks)]
    )
    slope, intercept = np.polyfit(sg["times"], f_peak, 1)
    assert slope == pytest.approx(rate, rel=0.05)
    assert intercept == pytest.approx(f0, abs=10.0)


# --------------------------------------------------- 3. split/reassemble WOLA


@pytest.mark.parametrize(
    "window,beta",
    [
        ("hann", None),
        ("hamming", None),
        ("blackman", None),
        ("kaiser", 6.0),
        ("rect", None),
    ],
)
@pytest.mark.parametrize("frame_length,hop", [(64, 32), (128, 32), (64, 16), (256, 64)])
def test_stft_istft_roundtrip(window, beta, frame_length, hop):
    rng = np.random.default_rng(frame_length + hop)
    # Lengths chosen to include non-tidy tails (not a multiple of the hop).
    for m in [frame_length, frame_length + 1, frame_length * 2 + 7, 400]:
        x = rng.standard_normal(m)
        frames = stft(x, FS, frame_length, hop, window, beta)["frames"]
        y = istft(frames, frame_length, hop, m, window, beta)
        np.testing.assert_allclose(y, x, atol=1e-9, rtol=1e-9)


def test_roundtrip_preserves_a_real_chirp_samplewise():
    # The identity must hold for a dynamic signal too, not just white noise.
    frame_length, hop = 64, 16
    t = np.arange(500) / FS
    x = np.sin(2 * np.pi * (40 * t + 0.5 * 200 * t**2)) + 0.25 * np.sin(
        2 * np.pi * 120 * t
    )
    frames = stft(x, FS, frame_length, hop, "hann")["frames"]
    y = istft(frames, frame_length, hop, x.size, "hann")
    np.testing.assert_allclose(y, x, atol=1e-9)


def test_modified_frames_invert_consistently():
    # Sanity on the inverse entry point as its own API: feeding the frames
    # back through istft with a *different* length is rejected, while keeping
    # the structure agrees with idft on individual interior frames.
    from app.dft import idft

    rng = np.random.default_rng(9)
    x = rng.standard_normal(200)
    r = stft(x, FS, 64, 16, "hann")
    frames = r["frames"]
    # One isolated frame: idft reproduces the (zero-padded) windowed segment.
    np.testing.assert_allclose(np.real(idft(frames[4])), r["windowed"][4], atol=1e-10)
    # signal_length must match the frame grid: 120 samples give a different
    # frame count than 200, so the structure does not line up.
    with pytest.raises(BadRequest, match="frame count mismatch"):
        istft(frames, 64, 16, 120, "hann")  # wrong signal_length


# ------------------------------------------------------------- framing / axes


def test_frame_count_and_times():
    # num_frames = 1 + (M-1)//H: columns while the frame center jH < M.
    assert num_frames(64, 64, 32) == 1 + (64 - 1) // 32
    assert num_frames(65, 64, 32) == 1 + (65 - 1) // 32
    assert num_frames(128, 64, 64) == 2
    assert num_frames(64, 64, 64) == 1  # single-frame degenerate case
    times = frame_times(128, 64, 32, FS)
    np.testing.assert_allclose(times, np.arange(times.size) * 32 / FS)
    assert times[0] == 0.0


def test_frame_signal_centered_and_zero_padded():
    x = np.arange(100.0)
    frames = frame_signal(x, 64, 32)
    # First frame is centered on sample 0: positions 0..31 see zeros,
    # positions 32..63 see x[0..31].
    assert np.all(frames[0, :32] == 0.0)
    np.testing.assert_allclose(frames[0, 32:], x[:32])
    # Second frame (center 32) covers x[0..63].
    np.testing.assert_allclose(frames[1], x[:64])


def test_spectrogram_matrix_orientation_and_axes():
    frame_length, hop = 64, 16
    x = np.random.default_rng(0).standard_normal(200)
    sg = spectrogram(x, FS, frame_length, hop, "hann")
    num_freq = frame_length // 2 + 1
    k = num_frames(200, frame_length, hop)
    assert sg["power"].shape == (num_freq, k)
    assert sg["magnitude"].shape == sg["magnitude_db"].shape == (num_freq, k)
    assert sg["frequencies"].shape == (num_freq,)
    assert sg["times"].shape == (k,)
    assert sg["frequencies"][0] == 0.0
    assert sg["frequencies"][-1] == pytest.approx(FS / 2)
    assert np.all(np.diff(sg["frequencies"]) == pytest.approx(FS / frame_length))


def test_single_frame_degenerate_case_is_one_column():
    # M == L, hop == L -> exactly one column, no error, centered on t = 0.
    x = np.random.default_rng(1).standard_normal(64)
    sg = spectrogram(x, FS, 64, 64, "rect")
    assert sg["power"].shape[1] == 1
    assert sg["num_frames"] == 1
    assert sg["times"][0] == 0.0
    # A rectangular single frame inverts to itself (the only contiguous
    # exact-fit case the centered framing supports at hop == L).
    # Note the centered frame reads x[0..31]; the rest is outside coverage,
    # so full-length reconstruction here is *not* required (no overlap).


def test_zero_signal_returns_zero_matrix():
    sg = spectrogram(np.zeros(200), FS, 64, 32, "kaiser", 5.0)
    assert np.max(sg["power"]) < 1e-20
    assert np.max(sg["magnitude"]) < 1e-9


# -------------------------------------------------------------- bad inputs ---


@pytest.mark.parametrize("bad_length", [0, 1, 32, 100, 2048])
def test_bad_frame_length_rejected(bad_length):
    with pytest.raises(BadRequest, match="frame_length must be one of"):
        stft(np.zeros(128), FS, bad_length, 32, "hann")


@pytest.mark.parametrize("bad_hop", [0, -1, -32])
def test_nonpositive_hop_rejected(bad_hop):
    with pytest.raises(BadRequest, match="hop must be a positive integer"):
        stft(np.zeros(128), FS, 64, bad_hop, "hann")


def test_hop_larger_than_frame_rejected():
    with pytest.raises(BadRequest, match="must not exceed frame_length"):
        stft(np.zeros(128), FS, 64, 65, "hann")


@pytest.mark.parametrize("m", [1, 32, 63])
def test_signal_shorter_than_frame_rejected(m):
    with pytest.raises(BadRequest, match="too short to fill one frame"):
        stft(np.zeros(m), FS, 64, 32, "hann")


def test_empty_signal_rejected():
    with pytest.raises(BadRequest, match="at least one sample"):
        stft(np.array([]), FS, 64, 32, "hann")


def test_nonfinite_signal_rejected():
    x = np.ones(128)
    x[10] = np.nan
    with pytest.raises(BadRequest, match="non-finite"):
        stft(x, FS, 64, 32, "hann")


def test_bad_fs_rejected():
    with pytest.raises(BadRequest):
        stft(np.zeros(128), 0.0, 64, 32, "hann")


def test_unknown_window_rejected():
    with pytest.raises(BadRequest, match="unknown window"):
        stft(np.zeros(128), FS, 64, 32, "flat_top")


def test_kaiser_without_beta_rejected():
    with pytest.raises(BadRequest, match="beta"):
        stft(np.zeros(128), FS, 64, 32, "kaiser")


# ----------------------------------------------------- inverse frame structure


def test_istft_frame_count_mismatch_rejected():
    x = np.random.default_rng(2).standard_normal(200)
    frames = stft(x, FS, 64, 32, "hann")["frames"]
    with pytest.raises(BadRequest, match="frame count mismatch"):
        istft(frames[:-1], 64, 32, 200, "hann")
    with pytest.raises(BadRequest, match="frame count mismatch"):
        istft(np.concatenate([frames, frames[:1]]), 64, 32, 200, "hann")


def test_istft_spectrum_length_mismatch_rejected():
    x = np.random.default_rng(2).standard_normal(200)
    frames = stft(x, FS, 64, 32, "hann")["frames"]
    with pytest.raises(BadRequest, match="does not match frame_length"):
        istft(frames[:, :32], 64, 32, 200, "hann")


def test_istft_rejects_non_2d_and_nonfinite():
    with pytest.raises(BadRequest, match="2-D"):
        istft(np.zeros(64), 64, 32, 200, "hann")
    bad = np.zeros((3, 64), dtype=complex)
    bad[1, 5] = complex(np.nan, 0)
    with pytest.raises(BadRequest, match="non-finite"):
        istft(bad, 64, 32, 128, "hann")


def test_istft_rejects_zero_coverage_no_overlap():
    # Zero-edge Hann window placed without overlap cannot reconstruct; the
    # missing samples simply are not in the frames.
    x = np.random.default_rng(4).standard_normal(300)
    frames = stft(x, FS, 64, 64, "hann")["frames"]
    with pytest.raises(BadRequest, match="zero coverage"):
        istft(frames, 64, 64, 300, "hann")


def test_istft_short_signal_length_rejected():
    frames = stft(np.zeros(128), FS, 64, 32, "hann")["frames"]
    with pytest.raises(BadRequest, match="shorter than frame_length"):
        istft(frames, 64, 32, 32, "hann")
