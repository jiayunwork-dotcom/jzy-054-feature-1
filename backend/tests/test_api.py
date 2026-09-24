"""End-to-end API tests through FastAPI's TestClient, including every
input-validation rule the product spec calls out.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------- DFT round trip through the API -----------------------------------------------


def test_dft_and_idft_roundtrip():
    x = list(np.sin(np.linspace(0, 6, 128)) + 0.3)
    r = client.post("/api/dft", json={"signal": x, "fs": 500, "n": 128})
    assert r.status_code == 200, r.text
    data = r.json()
    spec = data["spectra"][0]
    assert len(spec["real"]) == 128

    r2 = client.post(
        "/api/idft", json={"real": spec["real"], "imag": spec["imag"]}
    )
    assert r2.status_code == 200
    np.testing.assert_allclose(r2.json()["signal"], x, atol=1e-9)


def test_dft_multi_window_and_zero_padding():
    x = list(np.sin(2 * np.pi * 5 * np.arange(64) / 64))
    r = client.post(
        "/api/dft",
        json={
            "signal": x,
            "fs": 64,
            "n": 64,
            "padded_n": 256,
            "windows": [
                {"name": "rect"},
                {"name": "hann"},
                {"name": "hamming"},
                {"name": "blackman"},
                {"name": "kaiser", "beta": 6.0},
            ],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["n_padded"] == 256
    assert [s["window"] for s in data["spectra"]] == [
        "rect", "hann", "hamming", "blackman", "kaiser"
    ]
    for spec in data["spectra"]:
        assert len(spec["magnitude"]) == 256
        assert len(spec["power"]) == 256
        assert len(spec["phase"]) == 256


def test_zero_signal_is_zero_spectrum_not_an_error():
    r = client.post(
        "/api/dft",
        json={
            "signal": [0.0] * 64,
            "fs": 100,
            "n": 64,
            "windows": [{"name": "kaiser", "beta": 5}],
        },
    )
    assert r.status_code == 200
    assert max(r.json()["spectra"][0]["magnitude"]) < 1e-12


# ----------------------------------------------------------------- bad inputs ----------------------------------------------------------------


@pytest.mark.parametrize("bad_n", [32, 100, 2048, 0])
def test_bad_n_rejected(bad_n):
    r = client.post(
        "/api/dft", json={"signal": [1.0] * 16, "fs": 100, "n": bad_n}
    )
    assert r.status_code == 400
    assert "N must be" in r.json()["detail"]


def test_signal_longer_than_n_rejected():
    r = client.post(
        "/api/dft", json={"signal": [1.0] * 200, "fs": 100, "n": 128}
    )
    assert r.status_code == 400
    assert "exceeds" in r.json()["detail"]


@pytest.mark.parametrize("bad_fs", [0, -10])
def test_nonpositive_fs_rejected(bad_fs):
    r = client.post(
        "/api/dft", json={"signal": [1.0] * 16, "fs": bad_fs, "n": 64}
    )
    assert r.status_code == 400
    assert "sampling rate" in r.json()["detail"]


def test_unknown_window_rejected():
    r = client.post(
        "/api/dft",
        json={
            "signal": [1.0] * 16,
            "fs": 100,
            "n": 64,
            "windows": [{"name": "flat_top"}],
        },
    )
    assert r.status_code == 400
    assert "unknown window" in r.json()["detail"]


def test_kaiser_without_beta_rejected():
    r = client.post(
        "/api/dft",
        json={
            "signal": [1.0] * 16,
            "fs": 100,
            "n": 64,
            "windows": [{"name": "kaiser"}],
        },
    )
    assert r.status_code == 400
    assert "beta" in r.json()["detail"]


def test_padding_shorter_than_signal_rejected():
    r = client.post(
        "/api/dft",
        json={
            "signal": [1.0] * 128,
            "fs": 100,
            "n": 128,
            "padded_n": 64,
        },
    )
    assert r.status_code == 400
    assert "shorter" in r.json()["detail"]


def test_empty_signal_rejected():
    r = client.post("/api/dft", json={"signal": [], "fs": 100, "n": 64})
    assert r.status_code == 422 or r.status_code == 400


# ------------------------------------------------------------------- windows ------------------------------------------------------------------


def test_windows_endpoint_metrics():
    r = client.post(
        "/api/windows",
        json={
            "n": 128,
            "windows": [
                {"name": "rect"},
                {"name": "hann"},
                {"name": "hamming"},
                {"name": "blackman"},
                {"name": "kaiser", "beta": 8.6},
            ],
        },
    )
    assert r.status_code == 200
    infos = {w["name"]: w for w in r.json()["windows"]}
    assert len(infos["hann"]["coefficients"]) == 128
    assert infos["rect"]["mainlobe_bins"] == 2.0
    assert infos["hann"]["mainlobe_bins"] == 4.0
    assert infos["hamming"]["peak_sidelobe_db"] == -42.7
    assert infos["blackman"]["mainlobe_bins"] == 6.0
    kaiser = infos["kaiser"]
    assert kaiser["mainlobe_bins"] > 6
    assert kaiser["peak_sidelobe_db"] < -50


def test_windows_endpoint_bad_name():
    r = client.post(
        "/api/windows", json={"n": 32, "windows": [{"name": "nope"}]}
    )
    assert r.status_code == 400


# ------------------------------------------------------------------- filter -------------------------------------------------------------------


def test_filter_api_lowpass():
    n, fs = 256, 256.0
    t = np.arange(n) / fs
    x = np.cos(2 * np.pi * 10 * t) + np.cos(2 * np.pi * 90 * t)
    r = client.post(
        "/api/filter",
        json={
            "signal": list(x),
            "fs": fs,
            "mode": "lowpass",
            "cutoff_high": 30.0,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    y = np.asarray(data["filtered_signal"])
    np.testing.assert_allclose(y, np.cos(2 * np.pi * 10 * t), atol=1e-8)
    assert data["residual_high_energy"] < 1e-18
    assert sum(data["mask"]) > 0


def test_filter_api_bad_band():
    r = client.post(
        "/api/filter",
        json={
            "signal": [0.0] * 64,
            "fs": 100,
            "mode": "bandpass",
            "cutoff_low": 80,
            "cutoff_high": 20,
        },
    )
    assert r.status_code == 400
    assert "strictly greater" in r.json()["detail"]


def test_filter_api_band_out_of_nyquist():
    r = client.post(
        "/api/filter",
        json={
            "signal": [0.0] * 64,
            "fs": 100,
            "mode": "lowpass",
            "cutoff_high": 80,
        },
    )
    assert r.status_code == 400
    assert "Nyquist" in r.json()["detail"]


# ------------------------------------------------------------------ sampling ------------------------------------------------------------------


def test_sampling_api_alias_flag_and_apparent_freq():
    r = client.post(
        "/api/sampling", json={"signal_freq": 150, "fs": 200, "n_samples": 24}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["aliased"] is True
    assert data["apparent_freq_hz"] == pytest.approx(50.0)
    assert len(data["original_t"]) == 23 * 30 + 1
    assert len(data["sample_t"]) == 24


def test_sampling_api_no_alias():
    r = client.post(
        "/api/sampling", json={"signal_freq": 50, "fs": 200}
    )
    assert r.status_code == 200
    assert r.json()["aliased"] is False
    assert r.json()["apparent_freq_hz"] == pytest.approx(50.0)


# -------------------------------------------------------------- spectrogram ---------------------------------------------------------------


def test_stft_api_shape_and_resolutions():
    n, fs, frame, hop = 1024, 256.0, 256, 64
    t = np.arange(n) / fs
    x = list(np.sin(2 * np.pi * 20 * t))
    r = client.post(
        "/api/stft",
        json={
            "signal": x,
            "fs": fs,
            "frame_length": frame,
            "hop_length": hop,
            "window": {"name": "hann"},
        },
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["frequencies"]) == frame // 2 + 1
    assert d["frequencies"][1] == pytest.approx(fs / frame)
    assert d["frequencies"][-1] == pytest.approx(fs / 2)
    assert len(d["magnitude"]) == d["num_frames"]
    assert all(len(row) == frame // 2 + 1 for row in d["magnitude"])
    assert d["time_resolution_s"] == pytest.approx(frame / fs)
    assert d["frequency_resolution_hz"] == pytest.approx(fs / frame)
    assert d["overlap_ratio"] == pytest.approx(1 - hop / frame)
    assert d["spectra_real"] is None  # off by default -> payload stays light


def test_stft_tone_is_horizontal_line():
    fs = 256.0
    t = np.arange(1024) / fs
    body = {
        "signal": list(np.sin(2 * np.pi * 48 * t)),
        "fs": fs,
        "frame_length": 256,
        "hop_length": 64,
        "window": {"name": "hann"},
    }
    r = client.post("/api/stft", json=body)
    d = r.json()
    mag = np.asarray(d["magnitude"])
    starts = np.asarray(d["frame_starts"])
    contained = (starts >= 0) & (starts + 256 <= 1024)
    peaks = np.argmax(mag[contained], axis=1)
    assert set(np.unique(peaks)) == {48}  # 48 Hz = bin 48 at fs/N = 1 Hz


def test_stft_chirp_peaks_rise():
    fs = 256.0
    t = np.arange(2048) / fs
    x = np.sin(2 * np.pi * (8 * t + 0.5 * 11.5 * t**2))
    r = client.post(
        "/api/stft",
        json={
            "signal": list(x),
            "fs": fs,
            "frame_length": 256,
            "hop_length": 64,
            "window": {"name": "hann"},
        },
    )
    d = r.json()
    mag = np.asarray(d["magnitude"])
    starts = np.asarray(d["frame_starts"])
    contained = (starts >= 0) & (starts + 256 <= 2048)
    peaks = np.argmax(mag[contained], axis=1)
    assert np.all(np.diff(peaks) >= 0)
    assert peaks[-1] - peaks[0] >= 50


def test_stft_then_istft_api_roundtrip():
    rng = np.random.default_rng(11)
    x = list(rng.standard_normal(1024))
    r = client.post(
        "/api/stft",
        json={
            "signal": x,
            "fs": 256,
            "frame_length": 256,
            "hop_length": 128,
            "window": {"name": "hann"},
            "include_spectrum": True,
        },
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["spectra_real"]) == d["num_frames"]
    assert len(d["spectra_real"][0]) == 256
    r2 = client.post(
        "/api/istft",
        json={
            "fs": 256,
            "frame_length": 256,
            "hop_length": 128,
            "signal_length": 1024,
            "spectra_real": d["spectra_real"],
            "spectra_imag": d["spectra_imag"],
            "window": {"name": "hann"},
        },
    )
    assert r2.status_code == 200, r2.text
    np.testing.assert_allclose(r2.json()["signal"], x, atol=1e-9)


def test_stft_single_frame_signal_is_one_column():
    r = client.post(
        "/api/stft",
        json={
            "signal": [0.0] * 128,
            "fs": 100,
            "frame_length": 128,
            "hop_length": 64,
        },
    )
    assert r.status_code == 200
    assert r.json()["num_frames"] == 1
    assert len(r.json()["times"]) == 1


@pytest.mark.parametrize("bad_frame", [16, 63, 100, 1024])
def test_stft_bad_frame_length_rejected(bad_frame):
    r = client.post(
        "/api/stft",
        json={
            "signal": [0.0] * 256,
            "fs": 100,
            "frame_length": bad_frame,
            "hop_length": 32,
        },
    )
    assert r.status_code == 400
    assert "frame_length" in r.json()["detail"]


@pytest.mark.parametrize("bad_hop", [0, -5])
def test_stft_nonpositive_hop_rejected(bad_hop):
    r = client.post(
        "/api/stft",
        json={
            "signal": [0.0] * 256,
            "fs": 100,
            "frame_length": 128,
            "hop_length": bad_hop,
        },
    )
    assert r.status_code == 400
    assert "hop_length" in r.json()["detail"]


def test_stft_hop_larger_than_frame_rejected():
    r = client.post(
        "/api/stft",
        json={
            "signal": [0.0] * 256,
            "fs": 100,
            "frame_length": 128,
            "hop_length": 200,
        },
    )
    assert r.status_code == 400
    assert "must not exceed" in r.json()["detail"]


def test_stft_signal_shorter_than_frame_rejected():
    r = client.post(
        "/api/stft",
        json={
            "signal": [0.0] * 100,
            "fs": 100,
            "frame_length": 128,
            "hop_length": 64,
        },
    )
    assert r.status_code == 400
    assert "too short" in r.json()["detail"]


def test_istft_frame_bin_width_mismatch_rejected():
    r = client.post(
        "/api/istft",
        json={
            "fs": 100,
            "frame_length": 128,
            "hop_length": 64,
            "signal_length": 256,
            # Only the one-sided 65 bins supplied: structural mismatch.
            "spectra_real": [[0.0] * 65 for _ in range(4)],
            "spectra_imag": [[0.0] * 65 for _ in range(4)],
        },
    )
    assert r.status_code == 400
    assert "frame_length" in r.json()["detail"]


def test_istft_frame_count_mismatch_rejected():
    r = client.post(
        "/api/istft",
        json={
            "fs": 100,
            "frame_length": 128,
            "hop_length": 64,
            "signal_length": 1024,
            "spectra_real": [[0.0] * 128 for _ in range(2)],
            "spectra_imag": [[0.0] * 128 for _ in range(2)],
        },
    )
    assert r.status_code == 400
    assert "frame structure mismatch" in r.json()["detail"]
