"""Identity tests for the DFT/IDFT kernel.

These guard the four relationships the product spec requires to really hold:

1. forward then inverse transform reproduces the signal;
2. Parseval: sum|x|^2 == (1/N) sum|X|^2;
3. an on-bin pure tone has a two-spike magnitude spectrum;
4. real signals produce conjugate-symmetric spectra;
plus a cross-check of the hand-written FFT against numpy.fft.
"""

import numpy as np
import pytest

from app.dft import analyze, dft, dft_frequencies, idft
from app.windows import get_window, window_metrics

SIZES = [1, 2, 3, 5, 16, 64, 128, 256, 100, 1000]


@pytest.mark.parametrize("n", SIZES)
def test_inverse_roundtrip(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    x_hat = idft(dft(x))
    np.testing.assert_allclose(x_hat, x, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("n", [16, 64, 128, 256, 1024])
def test_parseval(n):
    rng = np.random.default_rng(42 + n)
    x = rng.standard_normal(n)
    xk = dft(x)
    time_energy = float(np.sum(np.abs(x) ** 2))
    freq_energy = float(np.sum(np.abs(xk) ** 2) / n)
    assert freq_energy == pytest.approx(time_energy, rel=1e-10)


@pytest.mark.parametrize("n", [64, 128, 256])
def test_real_input_conjugate_symmetry(n):
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)
    xk = dft(x)
    # X[k] == conj(X[N-k]) for k = 1 .. N-1
    for k in range(1, n):
        assert xk[k] == pytest.approx(np.conj(xk[n - k]), abs=1e-9)
    # DC and Nyquist bins are real.
    assert abs(xk[0].imag) < 1e-9
    if n % 2 == 0:
        assert abs(xk[n // 2].imag) < 1e-9


def test_on_bin_tone_two_spikes():
    n, fs = 256, 256.0
    k0 = 7
    x = 3.0 * np.sin(2 * np.pi * k0 * np.arange(n) / n + 0.37)
    xk = dft(x)
    mag = np.abs(xk)
    top = set(np.argsort(mag)[-2:])
    assert top == {k0, n - k0}
    expected = n * 3.0 / 2.0
    assert mag[k0] == pytest.approx(expected, rel=1e-9)
    assert mag[n - k0] == pytest.approx(expected, rel=1e-9)
    # Everywhere else is at numerical-noise level.
    other = np.delete(mag, [k0, n - k0])
    assert np.max(other) < 1e-8


def test_on_bin_cosine_dc_and_nyquist():
    # DC tone lands purely on bin 0; Nyquist tone purely on bin N/2.
    n = 128
    dc = np.full(n, 2.5)
    assert np.abs(dft(dc)[0]) == pytest.approx(n * 2.5)
    assert np.max(np.abs(np.delete(dft(dc), 0))) < 1e-8

    nyq = 1.5 * np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    xk = dft(nyq)
    assert np.abs(xk[n // 2]) == pytest.approx(n * 1.5)
    assert np.max(np.abs(np.delete(xk, n // 2))) < 1e-8


def test_matches_numpy_fft():
    pytest.importorskip("numpy.fft")
    rng = np.random.default_rng(123)
    for n in (64, 127, 256, 500):
        x = rng.standard_normal(n)
        np.testing.assert_allclose(dft(x), np.fft.fft(x), atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(idft(dft(x)), x, atol=1e-9)


def test_zero_signal_has_zero_spectrum():
    x = np.zeros(128)
    xk = analyze(x, window=get_window("hann", 128), n_padded=256)
    assert np.allclose(xk, 0.0)
    assert dft_frequencies(256, 1000.0)[1] == pytest.approx(1000.0 / 256.0)


def test_zero_padding_interpolates_spectrum():
    # Same signal: padded DFT is a denser sampling of the same DTFT, so its
    # non-zero-padded bin values must reappear at the corresponding k.
    n = 64
    x = np.sin(2 * np.pi * 5 * np.arange(n) / n)
    coarse = dft(x)
    fine = analyze(x, n_padded=256)
    for k in range(n):
        assert fine[4 * k] == pytest.approx(coarse[k], abs=1e-8)


def test_window_metrics_table_values():
    assert window_metrics("rect")["mainlobe_bins"] == 2.0
    assert window_metrics("hann")["mainlobe_bins"] == 4.0
    assert window_metrics("hamming")["mainlobe_bins"] == 4.0
    assert window_metrics("blackman")["mainlobe_bins"] == 6.0

    # Measured Kaiser figures should track beta: larger beta widens the main
    # lobe and pushes sidelobes down. beta=0 is the rectangular window.
    m0 = window_metrics("kaiser", 0.0)
    assert m0["peak_sidelobe_db"] == pytest.approx(-13.3, abs=1.0)
    m5 = window_metrics("kaiser", 5.0)
    m10 = window_metrics("kaiser", 10.0)
    assert m10["mainlobe_bins"] > m5["mainlobe_bins"] > m0["mainlobe_bins"]
    assert m10["peak_sidelobe_db"] < m5["peak_sidelobe_db"] < m0["peak_sidelobe_db"]
