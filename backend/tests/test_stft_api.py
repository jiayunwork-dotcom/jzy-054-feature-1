"""End-to-end HTTP tests for the short-time analysis endpoints.

Covers the spectrogram payload (matrix orientation, axes, resolution
figures), the STFT -> ISTFT round trip through JSON, and every illegal
input the product spec calls out.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

FS = 1000.0


# --------------------------------------------------------------------- happy


def _chirp(m=600, f0=30.0, rate=300.0):
    t = np.arange(m) / FS
    return list(np.sin(2 * np.pi * (f0 * t + 0.5 * rate * t**2)))


def test_stft_payload_shapes_and_resolution():
    x = list(np.random.default_rng(0).standard_normal(300))
    r = client.post(
        "/api/stft",
        json={
            "signal": x,
            "fs": FS,
            "frame_length": 64,
            "hop": 16,
            "window": {"name": "hann"},
        },
    )
    assert r.status_code == 200, r.text
    d = r.json()
    num_freq = 33  # 64/2 + 1
    num_frames = 1 + (300 - 1) // 16
    assert len(d["power"]) == num_freq
    assert all(len(row) == d["num_frames"] for row in d["power"])
    assert d["num_frames"] == num_frames
    assert len(d["frequencies"]) == num_freq
    assert len(d["times"]) == num_frames
    assert d["frequencies"][0] == 0.0
    assert d["frequencies"][-1] == pytest.approx(FS / 2)
    # dB values are defined for every cell (floored, never -inf/NaN).
    assert all(np.isfinite(v) for row in d["magnitude_db"] for v in row)
    assert min(min(row) for row in d["magnitude_db"]) >= -120.0
    # Resolution figures the UI surfaces for the trade-off readout.
    assert d["time_resolution_s"] == pytest.approx(16 / FS)
    assert d["frame_duration_s"] == pytest.approx(64 / FS)
    assert d["frequency_resolution_hz"] == pytest.approx(FS / 64)
    assert d["overlap_ratio"] == pytest.approx(0.75)


def test_stft_tone_is_a_horizontal_line():
    t = np.arange(400) / FS
    x = list(np.sin(2 * np.pi * 100 * t))
    d = client.post(
        "/api/stft",
        json={
            "signal": x,
            "fs": FS,
            "frame_length": 128,
            "hop": 32,
            "window": {"name": "hann"},
        },
    ).json()
    peaks = [int(np.argmax(col)) for col in np.array(d["power"]).T]
    assert set(peaks) == {peaks[0]}
    assert d["frequencies"][peaks[0]] == pytest.approx(100.0, abs=FS / 128)


def test_stft_chirp_sweeps_up():
    d = client.post(
        "/api/stft",
        json={
            "signal": _chirp(),
            "fs": FS,
            "frame_length": 128,
            "hop": 16,
            "window": {"name": "hann"},
        },
    ).json()
    peaks = np.argmax(np.array(d["power"]), axis=0)
    # The sweep moves several bins per column; peaks must rise over the body.
    assert peaks[-5] > peaks[2] + 5


def test_stft_istft_roundtrip_via_api():
    x = np.random.default_rng(7).standard_normal(300)
    d = client.post(
        "/api/stft",
        json={
            "signal": list(x),
            "fs": FS,
            "frame_length": 64,
            "hop": 32,
            "window": {"name": "kaiser", "beta": 6.0},
        },
    ).json()
    # Recompute the full complex frames to feed the inverse endpoint (the
    # spectrogram payload intentionally carries one-sided intensities only).
    from app.stft import stft

    frames = stft(x, FS, 64, 32, "kaiser", 6.0)["frames"]
    r = client.post(
        "/api/istft",
        json={
            "real": frames.real.tolist(),
            "imag": frames.imag.tolist(),
            "frame_length": 64,
            "hop": 32,
            "signal_length": 300,
            "window": {"name": "kaiser", "beta": 6.0},
        },
    )
    assert r.status_code == 200, r.text
    y = np.asarray(r.json()["signal"])
    np.testing.assert_allclose(y, x, atol=1e-9)
    assert r.json()["num_frames"] == d["num_frames"]


def test_single_frame_degenerate_case():
    # Exactly one frame -> one column, no error.
    x = list(np.random.default_rng(2).standard_normal(64))
    r = client.post(
        "/api/stft",
        json={
            "signal": x,
            "fs": FS,
            "frame_length": 64,
            "hop": 64,
            "window": {"name": "rect"},
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["num_frames"] == 1


def test_zero_signal_is_zero_matrix():
    r = client.post(
        "/api/stft",
        json={
            "signal": [0.0] * 200,
            "fs": FS,
            "frame_length": 64,
            "hop": 32,
            "window": {"name": "hann"},
        },
    )
    assert r.status_code == 200
    assert max(max(row) for row in r.json()["power"]) < 1e-20


# ------------------------------------------------------------------ bad input


@pytest.mark.parametrize("bad_length", [32, 100, 2048])
def test_bad_frame_length(bad_length):
    r = client.post(
        "/api/stft",
        json={
            "signal": [0.0] * 200,
            "fs": FS,
            "frame_length": bad_length,
            "hop": 32,
            "window": {"name": "hann"},
        },
    )
    assert r.status_code == 400
    assert "frame_length must be one of" in r.json()["detail"]


@pytest.mark.parametrize("bad_hop", [0, -8])
def test_nonpositive_hop(bad_hop):
    r = client.post(
        "/api/stft",
        json={
            "signal": [0.0] * 200,
            "fs": FS,
            "frame_length": 64,
            "hop": bad_hop,
            "window": {"name": "hann"},
        },
    )
    assert r.status_code == 400
    assert "hop" in r.json()["detail"]


def test_hop_exceeds_frame():
    r = client.post(
        "/api/stft",
        json={
            "signal": [0.0] * 200,
            "fs": FS,
            "frame_length": 64,
            "hop": 65,
            "window": {"name": "hann"},
        },
    )
    assert r.status_code == 400
    assert "must not exceed frame_length" in r.json()["detail"]


def test_signal_too_short():
    r = client.post(
        "/api/stft",
        json={
            "signal": [0.0] * 50,
            "fs": FS,
            "frame_length": 64,
            "hop": 32,
            "window": {"name": "hann"},
        },
    )
    assert r.status_code == 400
    assert "too short to fill one frame" in r.json()["detail"]


def test_stft_bad_fs():
    r = client.post(
        "/api/stft",
        json={
            "signal": [0.0] * 200,
            "fs": -1,
            "frame_length": 64,
            "hop": 32,
            "window": {"name": "hann"},
        },
    )
    assert r.status_code == 400


def test_istft_frame_count_mismatch():
    x = np.random.default_rng(5).standard_normal(200)
    from app.stft import stft

    frames = stft(x, FS, 64, 32, "hann")["frames"]
    r = client.post(
        "/api/istft",
        json={
            "real": frames[:-1].real.tolist(),
            "imag": frames[:-1].imag.tolist(),
            "frame_length": 64,
            "hop": 32,
            "signal_length": 200,
            "window": {"name": "hann"},
        },
    )
    assert r.status_code == 400
    assert "frame count mismatch" in r.json()["detail"]


def test_istft_spectrum_length_mismatch():
    x = np.random.default_rng(5).standard_normal(200)
    from app.stft import stft

    frames = stft(x, FS, 64, 32, "hann")["frames"]
    r = client.post(
        "/api/istft",
        json={
            "real": frames[:, :32].real.tolist(),
            "imag": frames[:, :32].imag.tolist(),
            "frame_length": 64,
            "hop": 32,
            "signal_length": 200,
            "window": {"name": "hann"},
        },
    )
    assert r.status_code == 400
    assert "does not match frame_length" in r.json()["detail"]
