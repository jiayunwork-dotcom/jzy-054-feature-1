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
