"""Identity tests for the short-time analysis module.

These guard the three time-frequency relationships the product spec calls
out, plus the framing geometry and every illegal-input rule:

1. a constant-frequency tone -> every frame's energy sits on the same
   frequency row (a horizontal line, no wandering);
2. a linear chirp -> the peak frequency row rises monotonically with the
   frame index (a diagonal judged by the machine, not by eye);
3. split -> IDFT -> weighted overlap-add recovers the original signal to
   numerical precision for COLA window/hop pairs (Hann 50 % overlap and
   friends);
plus axis coordinates, the single-column degenerate case and all 400s.
"""

import numpy as np
import pytest

from app.config import ALLOWED_FRAME_N
from app.dft import dft_frequencies
from app.spectrogram import (
    check_frame_params,
    extract_frames,
    frame_starts,
    istft,
    stft,
)
from app.errors import BadRequest


# ------------------------------------------------------------------- framing


@pytest.mark.parametrize("frame_length", sorted(ALLOWED_FRAME_N))
def test_frame_length_gears_accepted(frame_length):
    check_frame_params(frame_length, frame_length // 2)


@pytest.mark.parametrize("bad_n", [1, 16, 33, 63, 100, 1024])
def test_frame_length_outside_gears_rejected(bad_n):
    with pytest.raises(BadRequest, match="frame_length"):
        check_frame_params(bad_n, 16)


@pytest.mark.parametrize("bad_hop", [0, -4])
def test_nonpositive_hop_rejected(bad_hop):
    with pytest.raises(BadRequest, match="hop_length must be positive"):
        check_frame_params(128, bad_hop)


def test_hop_larger_than_frame_rejected():
    with pytest.raises(BadRequest, match="must not exceed"):
        check_frame_params(128, 129)


def test_signal_shorter_than_one_frame_rejected():
    with pytest.raises(BadRequest, match="too short"):
        frame_starts(127, 128, 64)


def test_single_frame_signal_is_one_column():
    # The legal degenerate case: signal exactly fills one frame.
    for hop in (1, 32, 64, 128):
        starts = frame_starts(128, 128, hop)
        assert starts == [0]
    result = stft(np.zeros(128), 128, 64, "hann")
    assert result["frames"].shape[0] == 1
    assert result["times"][0] == pytest.approx(64.0)


def test_frame_starts_are_hop_spaced_and_cover_the_signal():
    # Longer signal: interior starts exactly H apart, run from 0 while the
    # whole frame fits; head and tail frames bracket the interior set so
    # every sample is represented.
    length, frame_length, hop = 1000, 128, 64
    starts = np.asarray(frame_starts(length, frame_length, hop))
    interior = starts[(starts >= 0) & (starts <= length - frame_length)]
    assert np.all(np.diff(interior) == hop)
    assert interior[0] == 0
    assert interior[-1] == (length - frame_length) // hop * hop
    # A head context frame precedes the interior set and the last frame still
    # overlaps the signal (start < L).
    assert starts[0] == -(frame_length // 2)
    assert starts[-1] < length
    # Every sample lies inside at least one frame window.
    covered = np.zeros(length, dtype=bool)
    for s in starts:
        covered[max(0, s) : min(length, s + frame_length)] = True
    assert covered.all()


def test_extract_frames_zero_pads_context_frames():
    x = np.arange(10, dtype=np.float64)
    frames = extract_frames(x, 4, 2)  # L > N; head start -2, tail after 6
    assert frames.shape[1] == 4
    assert frames.shape[0] == len(frame_starts(10, 4, 2))
    # Interior frame at start 0 must equal x[0:4] verbatim.
    starts = frame_starts(10, 4, 2)
    col = starts.index(0)
    np.testing.assert_array_equal(frames[col], x[0:4])


# ------------------------------------------------------ relation 1: tone line


def test_constant_tone_is_a_horizontal_line():
    fs = 256.0
    n, freq = 512, 32.0
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * freq * t + 0.4)
    result = stft(x, 128, 32, "hann")
    half = 128 // 2 + 1
    mag = np.abs(result["frames"][:, :half])

    # Interior (fully real-signal) frames: identical peak row and the peak
    # row is the bin nearest 32 Hz = bin 16.
    interior = np.asarray(result["starts"]) >= 0
    peaks = np.argmax(mag[interior], axis=1)
    assert set(np.unique(peaks)) == {16}
    freqs = dft_frequencies(128, fs)[:half]
    assert freqs[16] == pytest.approx(freq)

    # Energy concentration: for frames fully contained in the signal (no
    # zero-padded overhang at either edge), at least 99.9 % of each frame's
    # one-sided spectral energy lies in the peak row and its ±2 neighbors
    # (the Hann main lobe is 4 bins wide).
    power = mag**2
    n = x.shape[0]
    contained = np.where(
        (np.asarray(result["starts"]) >= 0)
        & (np.asarray(result["starts"]) + 128 <= n)
    )[0]
    for col in contained:
        p = peaks[0]
        local = power[col, p - 2 : p + 3].sum()
        assert local / power[col].sum() > 0.999


@pytest.mark.parametrize("window_name", ["rect", "hann", "hamming", "blackman"])
def test_tone_peak_row_stable_across_windows(window_name):
    fs = 128.0
    t = np.arange(512) / fs
    x = np.cos(2 * np.pi * 16.0 * t)
    result = stft(x, 128, 32, window_name)
    interior = np.asarray(result["starts"]) >= 0
    peaks = np.argmax(np.abs(result["frames"][interior, :65]), axis=1)
    assert set(np.unique(peaks)) == {16}


# -------------------------------------------------- relation 2: chirp diagonal


def test_linear_chirp_peak_row_monotonically_rises():
    fs = 256.0
    n = 2048
    t = np.arange(n) / fs
    f0, slope = 8.0, 11.5  # Hz, Hz/s -> ends at 100 Hz, comfortably < fs/2
    phase = 2 * np.pi * (f0 * t + 0.5 * slope * t**2)
    x = np.sin(phase)

    result = stft(x, 256, 64, "hann")
    half = 256 // 2 + 1
    starts = np.asarray(result["starts"])
    # Judge on frames fully contained in the signal (no zero-padded context).
    contained = (starts >= 0) & (starts + 256 <= n)
    peaks = np.argmax(np.abs(result["frames"][contained, :half]), axis=1)

    # Strictly non-decreasing peak row ...
    assert np.all(np.diff(peaks) >= 0)
    # ... with a clear overall climb, not a flat accidental plateau.
    assert peaks[-1] - peaks[0] >= 50
    # The climb is a straight line: peak bin vs frame index is highly linear.
    corr = np.corrcoef(np.arange(peaks.size), peaks)[0, 1]
    assert corr > 0.995

    # And the swept rate agrees with the physical slope: row width is
    # fs/N = 1 Hz, column spacing H/fs = 0.25 s, so bins rise by ~2.875/col.
    measured_slope = (peaks[-1] - peaks[0]) / (peaks.size - 1) * (fs / 256) / (
        64 / fs
    )
    assert measured_slope == pytest.approx(slope, rel=0.12)


def test_chirp_peak_frequencies_track_the_signal():
    fs = 200.0
    n = 1000
    t = np.arange(n) / fs
    f0, slope = 10.0, 10.0
    x = np.sin(2 * np.pi * (f0 * t + 0.5 * slope * t**2))
    result = stft(x, 128, 32, "hann")
    freqs = dft_frequencies(128, fs)[:65]
    starts = np.asarray(result["starts"])
    contained = (starts >= 0) & (starts + 128 <= n)
    peaks = np.argmax(np.abs(result["frames"][contained, :65]), axis=1)
    peak_f = freqs[peaks]
    center_t = (starts[contained] + 64) / fs
    expected = f0 + slope * center_t
    # One-bin (1.5625 Hz) quantization tolerance around the ideal line.
    np.testing.assert_allclose(peak_f, expected, atol=2.0)


# ---------------------------------------------- relation 3: split/reassemble


@pytest.mark.parametrize(
    "frame_length,hop,window_name,kwargs",
    [
        (64, 32, "hann", {}),          # 50 % overlap, textbook COLA
        (128, 32, "hann", {}),         # 75 % overlap
        (256, 32, "hann", {}),         # 87.5 % overlap
        (128, 128, "rect", {}),        # no overlap, rectangular
        (256, 64, "blackman", {}),     # 75 % overlap
        (128, 64, "kaiser", {"beta": 6.0}),
        (128, 32, "hamming", {}),
    ],
)
def test_windowed_frames_reassemble_to_original(
    frame_length, hop, window_name, kwargs
):
    rng = np.random.default_rng(frame_length + hop)
    x = rng.standard_normal(1024)
    result = stft(x, frame_length, hop, window_name, **kwargs)
    y = istft(
        result["frames"],
        frame_length,
        hop,
        result["window"],
        signal_length=1024,
        starts=result["starts"],
    )
    np.testing.assert_allclose(y, x, atol=1e-10, rtol=1e-10)


def test_roundtrip_holds_for_awkward_non_tiling_length():
    rng = np.random.default_rng(9)
    for length in (129, 193, 300, 1001):
        x = rng.standard_normal(length)
        result = stft(x, 128, 64, "hann")
        y = istft(
            result["frames"],
            128,
            64,
            result["window"],
            signal_length=length,
            starts=result["starts"],
        )
        np.testing.assert_allclose(y, x, atol=1e-10)


def test_istft_infers_frame_positions():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(1024)
    result = stft(x, 256, 128, "hann")
    # No explicit starts: istft rebuilds the same grid from the parameters.
    y = istft(result["frames"], 256, 128, "hann", signal_length=1024)
    np.testing.assert_allclose(y, x, atol=1e-10)


def test_single_rectangular_frame_roundtrip():
    x = np.linspace(-1, 1, 128)
    result = stft(x, 128, 128, "rect")
    y = istft(
        result["frames"], 128, 128, "rect", signal_length=128, starts=result["starts"]
    )
    np.testing.assert_allclose(y, x, atol=1e-12)


# ------------------------------------------------------------- invalid frames


def test_istft_rejects_one_sided_matrix():
    bad = np.zeros((4, 65), dtype=np.complex128)
    with pytest.raises(BadRequest, match="frame_length"):
        istft(bad, 128, 64, "hann", signal_length=320)


def test_istft_rejects_frame_count_mismatch():
    bad = np.zeros((3, 128), dtype=np.complex128)
    with pytest.raises(BadRequest, match="frame structure mismatch"):
        istft(bad, 128, 64, "hann", signal_length=320)  # grid implies 7 frames


def test_istft_rejects_starts_count_mismatch():
    frames = np.zeros((4, 128), dtype=np.complex128)
    with pytest.raises(BadRequest, match="frame structure mismatch"):
        istft(
            frames,
            128,
            64,
            "hann",
            signal_length=320,
            starts=np.array([0, 64, 128, 192, 256]),
        )


def test_istft_rejects_empty_and_nonfinite_frames():
    with pytest.raises(BadRequest):
        istft(np.zeros((0, 8), dtype=np.complex128), 64, 32, "hann")
    bad = np.full((4, 64), np.inf + 0j)
    with pytest.raises(BadRequest, match="non-finite"):
        istft(bad, 64, 32, "hann", signal_length=256)


def test_stft_rejects_nonfinite_signal():
    x = np.zeros(128)
    x[5] = np.nan
    with pytest.raises(BadRequest, match="non-finite"):
        stft(x, 64, 32, "hann")


def test_unknown_window_rejected():
    with pytest.raises(BadRequest, match="unknown window"):
        stft(np.zeros(256), 128, 64, "no-such-window")


# ---------------------------------------------------------------- coordinates


def test_time_and_frequency_axes_are_physical_units():
    fs = 500.0
    n, hop = 256, 64
    result = stft(np.zeros(1024), n, hop, "hann")
    times = result["times"] / fs
    # First interior frame centered on t = 0; columns spaced H/fs = 0.128 s.
    interior = np.asarray(result["starts"]) >= 0
    interior_t = times[interior]
    np.testing.assert_allclose(np.diff(interior_t), hop / fs)
    assert interior_t[0] == pytest.approx((n / 2) / fs)
    freqs = dft_frequencies(n, fs)[: n // 2 + 1]
    assert freqs[1] == pytest.approx(fs / n)
    assert freqs[-1] == pytest.approx(fs / 2)
