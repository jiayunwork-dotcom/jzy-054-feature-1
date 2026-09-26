"""Identity tests for short-time Fourier analysis (app.stft) and its API.

These guard the three relationships the product spec calls out, plus every
validation rule:

1. a constant-frequency tone cuts into frames whose energy stays on the
   *same* frequency row in every frame (a horizontal line, no drifting);
2. a linear chirp's peak row rises monotonically with the frame index, at
   the slope the sweep rate implies;
3. with proper overlap and windowing, framing -> per-frame DFT -> weighted
   overlap-add ISTFT reconstructs the original signal to numerical
   precision (the round-trip identity);
4. illegal inputs (bad frame length, non-positive or oversized hop,
   shorter-than-one-frame signals, inconsistent frame matrices) are
   rejected with explanatory messages, while the legal degenerate case
   (signal == exactly one frame) returns a one-column spectrogram.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.errors import BadRequest
from app.main import app
from app.stft import check_frame_settings, frame_signal, istft, spectrogram, stft_frames

client = TestClient(app)

FS = 1024.0


def _tone(n, freq, fs=FS, amp=1.0, phase=0.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq * t + phase)


def _chirp(n, f0, f1, fs=FS):
    t = np.arange(n) / fs
    rate = (f1 - f0) / (n / fs)
    return np.sin(2 * np.pi * (f0 * t + 0.5 * rate * t * t))


def _peak_rows(magnitude):
    return np.argmax(magnitude, axis=1)


# ---------------------------------------------------------------
# relationship 1: constant tone -> same frequency row in every frame
# ---------------------------------------------------------------


def test_constant_tone_stays_on_one_row_rect():
    n, frame_len, hop = 1024, 128, 32
    k0 = 10  # on-bin frequency: 10 * fs / frame_len = 80 Hz
    x = _tone(n, k0 * FS / frame_len, amp=2.0)
    out = spectrogram(x, FS, frame_len, hop, "rect")
    mag = out["magnitude"]

    peaks = _peak_rows(mag)
    # Every single frame peaks on the very same row: a horizontal line.
    assert set(peaks.tolist()) == {k0}
    # And for a rectangular window the energy is *only* on that row.
    for j in range(mag.shape[0]):
        row_energy = float(np.sum(mag[j] ** 2))
        assert mag[j, k0] ** 2 == pytest.approx(row_energy, rel=1e-6)
    # Amplitude normalization: a 2.0-amplitude tone reads 2.0 at the peak.
    assert np.all(mag[:, k0] == pytest.approx(2.0, rel=1e-6))


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman"])
def test_constant_tone_stays_on_one_row_tapered(window):
    n, frame_len, hop = 1024, 128, 32
    k0 = 10
    x = _tone(n, k0 * FS / frame_len, amp=1.5)
    out = spectrogram(x, FS, frame_len, hop, window)
    peaks = _peak_rows(out["magnitude"])
    # Tapered windows smear the line over a few rows, but the peak row must
    # be identical in every frame — the line must not wander.
    assert set(peaks.tolist()) == {k0}
    assert np.all(out["magnitude"][:, k0] == pytest.approx(1.5, rel=0.02))


def test_kaiser_tone_peak_row():
    n, frame_len, hop = 512, 128, 64
    k0 = 7
    x = _tone(n, k0 * FS / frame_len)
    out = spectrogram(x, FS, frame_len, hop, "kaiser", beta=6.0)
    assert set(_peak_rows(out["magnitude"]).tolist()) == {k0}


# ---------------------------------------------------------------
# relationship 2: linear chirp -> monotonically rising ridge
# ---------------------------------------------------------------


def test_chirp_ridge_rises_monotonically_at_sweep_rate():
    n, frame_len, hop = 1024, 128, 32
    f0, f1 = 64.0, 448.0
    x = _chirp(n, f0, f1)
    out = spectrogram(x, FS, frame_len, hop, "hann")
    mag = out["magnitude"]
    peaks = _peak_rows(mag)
    df = out["freq_step_hz"]

    # The peak row never moves backwards as time advances.
    assert np.all(np.diff(peaks) >= 0)
    # And it climbs by roughly the expected number of rows.
    assert peaks[-1] - peaks[0] >= 30

    # Each frame's peak row tracks the instantaneous frequency at the
    # frame's center time to within a couple of bins.
    duration = n / FS
    rate = (f1 - f0) / duration  # Hz per second
    for m in range(len(peaks)):
        f_inst = f0 + rate * out["times"][m]
        assert abs(peaks[m] * df - f_inst) <= 2.0 * df

    # The ridge is straight: fitted slope matches the sweep rate converted
    # to bins per frame (rate [Hz/s] * hop [s/frame] / df [Hz/bin]).
    slope, _intercept = np.polyfit(np.arange(len(peaks)), peaks, 1)
    expected_slope = rate * (hop / FS) / df
    assert slope == pytest.approx(expected_slope, abs=0.15)


def test_chirp_ridge_direction_is_not_reversed():
    # A downward sweep must fall monotonically: the axis orientation is
    # guarded, not just "some diagonal exists".
    n, frame_len, hop = 1024, 128, 32
    x = _chirp(n, 448.0, 64.0)
    out = spectrogram(x, FS, frame_len, hop, "hann")
    peaks = _peak_rows(out["magnitude"])
    assert np.all(np.diff(peaks) <= 0)
    assert peaks[0] - peaks[-1] >= 30


# ---------------------------------------------------------------
# relationship 3: frame -> transform -> overlap-add reconstructs exactly
# ---------------------------------------------------------------


@pytest.mark.parametrize(
    "window,beta,hop",
    [
        ("rect", None, 128),  # no overlap at all: exact for the boxcar
        ("rect", None, 64),
        ("hann", None, 64),
        ("hann", None, 32),
        ("hamming", None, 64),
        ("blackman", None, 32),
        ("kaiser", 5.0, 64),
        ("kaiser", 8.6, 16),
    ],
)
def test_roundtrip_reconstructs_signal(window, beta, hop):
    frame_len = 128
    rng = np.random.default_rng(20240917)
    n = 512
    x = rng.standard_normal(n) + _tone(n, 100.0) + _tone(n, 230.0, amp=0.5)

    spectra, _starts = stft_frames(
        x, frame_len, hop, window, beta, pad_left=frame_len // 2
    )
    # Padded length n + frame_len; n is a multiple of every hop tested here.
    assert spectra.shape == (n // hop + 1, frame_len)

    recon = istft(spectra, hop, window, beta, pad_left=frame_len // 2, signal_length=n)
    np.testing.assert_allclose(recon, x, atol=1e-9)


def test_roundtrip_single_frame_signal():
    # A signal exactly one frame long still round-trips (through padding).
    frame_len, hop = 128, 64
    x = _tone(frame_len, 90.0) + 0.3
    spectra, _ = stft_frames(x, frame_len, hop, "hann", pad_left=frame_len // 2)
    recon = istft(spectra, hop, "hann", pad_left=frame_len // 2, signal_length=frame_len)
    np.testing.assert_allclose(recon, x, atol=1e-9)


def test_roundtrip_through_api():
    frame_len, hop, n = 128, 32, 512
    x = (_tone(n, 100.0) + 0.4 * _tone(n, 301.0)).tolist()
    r = client.post(
        "/api/stft",
        json={
            "signal": x,
            "fs": FS,
            "frame_len": frame_len,
            "hop": hop,
            "window": {"name": "hann"},
            "pad_left": frame_len // 2,
            "include_frames": True,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    r2 = client.post(
        "/api/istft",
        json={
            "frames_real": data["frames_real"],
            "frames_imag": data["frames_imag"],
            "hop": hop,
            "window": {"name": "hann"},
            "pad_left": frame_len // 2,
            "signal_length": n,
        },
    )
    assert r2.status_code == 200, r2.text
    np.testing.assert_allclose(r2.json()["signal"], x, atol=1e-9)


def test_zero_weight_overlap_rejected():
    # Hann window with hop == frame_len: window zeros coincide with
    # uncovered samples, so exact reconstruction is impossible and the
    # module must say so instead of dividing by zero.
    frame_len = 128
    x = _tone(512, 100.0)
    spectra, _ = stft_frames(x, frame_len, frame_len, "hann", pad_left=frame_len // 2)
    with pytest.raises(BadRequest, match="zero reconstruction weight"):
        istft(spectra, frame_len, "hann", pad_left=frame_len // 2, signal_length=512)


# ---------------------------------------------------------------
# axes, shapes and the legal degenerate single-frame case
# ---------------------------------------------------------------


def test_spectrogram_axes_and_resolutions():
    n, frame_len, hop = 1024, 256, 64
    x = _tone(n, 100.0)
    out = spectrogram(x, FS, frame_len, hop, "hann")
    assert out["num_frames"] == 1 + (n - frame_len) // hop
    assert out["num_bins"] == frame_len // 2 + 1
    assert out["magnitude"].shape == (out["num_frames"], out["num_bins"])

    freqs = out["frequencies"]
    assert freqs[0] == 0.0
    assert freqs[-1] == pytest.approx(FS / 2)
    assert np.allclose(np.diff(freqs), FS / frame_len)

    times = out["times"]
    assert times[0] == pytest.approx(frame_len / 2 / FS)
    assert np.allclose(np.diff(times), hop / FS)

    assert out["frame_duration_s"] == pytest.approx(frame_len / FS)
    assert out["time_step_s"] == pytest.approx(hop / FS)
    assert out["freq_step_hz"] == pytest.approx(FS / frame_len)


def test_single_frame_degenerate_via_api():
    x = _tone(64, 100.0).tolist()
    r = client.post(
        "/api/stft",
        json={"signal": x, "fs": FS, "frame_len": 64, "hop": 16},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["num_frames"] == 1
    assert len(data["magnitude"]) == 1
    assert len(data["magnitude"][0]) == 33
    assert len(data["times"]) == 1
    assert data["times"][0] == pytest.approx(32.0 / FS)


def test_frame_signal_starts_and_shape():
    x = np.arange(300, dtype=np.float64)
    frames, starts = frame_signal(x, 128, 32)
    # Only complete frames of the raw signal: 1 + (300 - 128) // 32.
    assert starts.tolist() == [0, 32, 64, 96, 128, 160]
    assert frames.shape == (6, 128)
    np.testing.assert_array_equal(frames[2], x[64:192])


# ---------------------------------------------------------------
# invalid inputs: module level
# ---------------------------------------------------------------


@pytest.mark.parametrize("bad_len", [0, 32, 100, 2048, -64])
def test_bad_frame_length_rejected(bad_len):
    with pytest.raises(BadRequest, match="frame length must be one of"):
        check_frame_settings(bad_len, 1)


@pytest.mark.parametrize("bad_hop", [0, -1, -64])
def test_nonpositive_hop_rejected(bad_hop):
    with pytest.raises(BadRequest, match="hop size must be positive"):
        check_frame_settings(128, bad_hop)


def test_hop_larger_than_frame_rejected():
    with pytest.raises(BadRequest, match="must not exceed the frame length"):
        check_frame_settings(64, 65)


def test_signal_shorter_than_one_frame_rejected():
    with pytest.raises(BadRequest, match="shorter than one frame"):
        frame_signal(np.zeros(100), 128, 32)


def test_istft_frame_structure_mismatch():
    spectra, _ = stft_frames(np.zeros(512), 128, 32, "hann", pad_left=64)
    # Claiming more samples than the frames can cover.
    with pytest.raises(BadRequest, match="inconsistent with signal_length"):
        istft(spectra, 32, "hann", pad_left=64, signal_length=10_000)


# ---------------------------------------------------------------
# invalid inputs: API level
# ---------------------------------------------------------------


def _stft_payload(**overrides):
    payload = {
        "signal": _tone(512, 100.0).tolist(),
        "fs": FS,
        "frame_len": 128,
        "hop": 32,
        "window": {"name": "hann"},
    }
    payload.update(overrides)
    return payload


def test_api_bad_frame_length():
    r = client.post("/api/stft", json=_stft_payload(frame_len=100))
    assert r.status_code == 400
    assert "frame length must be one of" in r.json()["detail"]


def test_api_nonpositive_hop():
    r = client.post("/api/stft", json=_stft_payload(hop=0))
    assert r.status_code == 400
    assert "hop size must be positive" in r.json()["detail"]


def test_api_hop_exceeds_frame_length():
    r = client.post("/api/stft", json=_stft_payload(frame_len=64, hop=65))
    assert r.status_code == 400
    assert "must not exceed the frame length" in r.json()["detail"]


def test_api_signal_shorter_than_frame():
    r = client.post(
        "/api/stft",
        json=_stft_payload(signal=[1.0] * 100, frame_len=128, hop=32),
    )
    assert r.status_code == 400
    assert "shorter than one frame" in r.json()["detail"]


def test_api_unknown_window():
    r = client.post("/api/stft", json=_stft_payload(window={"name": "flat_top"}))
    assert r.status_code == 400
    assert "unknown window" in r.json()["detail"]


def test_api_kaiser_without_beta():
    r = client.post("/api/stft", json=_stft_payload(window={"name": "kaiser"}))
    assert r.status_code == 400
    assert "beta" in r.json()["detail"]


def test_api_istft_ragged_frames_rejected():
    r = client.post(
        "/api/istft",
        json={
            "frames_real": [[1.0, 2.0, 3.0], [4.0, 5.0]],
            "hop": 1,
            "signal_length": 4,
        },
    )
    assert r.status_code == 400
    assert "inconsistent lengths" in r.json()["detail"]


def test_api_istft_imag_shape_mismatch_rejected():
    r = client.post(
        "/api/istft",
        json={
            "frames_real": [[1.0] * 64, [2.0] * 64],
            "frames_imag": [[0.0] * 64],
            "hop": 32,
            "signal_length": 64,
        },
    )
    assert r.status_code == 400
    assert "mismatch" in r.json()["detail"]


def test_api_istft_bad_frame_length_rejected():
    r = client.post(
        "/api/istft",
        json={"frames_real": [[1.0] * 100], "hop": 1, "signal_length": 100},
    )
    assert r.status_code == 400
    assert "frame length must be one of" in r.json()["detail"]


def test_api_istft_coverage_mismatch_rejected():
    frame_len, hop, n = 128, 32, 512
    x = _tone(n, 100.0)
    spectra, _ = stft_frames(x, frame_len, hop, "hann", pad_left=frame_len // 2)
    r = client.post(
        "/api/istft",
        json={
            "frames_real": spectra.real.tolist(),
            "frames_imag": spectra.imag.tolist(),
            "hop": hop,
            "window": {"name": "hann"},
            "pad_left": frame_len // 2,
            "signal_length": n + 500,  # frames cannot cover this many samples
        },
    )
    assert r.status_code == 400
    assert "inconsistent with signal_length" in r.json()["detail"]


def test_api_istft_zero_weight_rejected():
    frame_len = 128
    x = _tone(512, 100.0)
    spectra, _ = stft_frames(x, frame_len, frame_len, "hann", pad_left=frame_len // 2)
    r = client.post(
        "/api/istft",
        json={
            "frames_real": spectra.real.tolist(),
            "frames_imag": spectra.imag.tolist(),
            "hop": frame_len,
            "window": {"name": "hann"},
            "pad_left": frame_len // 2,
            "signal_length": 512,
        },
    )
    assert r.status_code == 400
    assert "zero reconstruction weight" in r.json()["detail"]
