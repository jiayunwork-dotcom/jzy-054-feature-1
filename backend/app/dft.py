"""Discrete Fourier transform kernel.

Conventions (matching numpy's FFT, so students familiar with textbooks that
use the same normalization can compare results):

.. math::

    X[k]   = \\sum_{n=0}^{N-1} x[n]\\, e^{-j 2\\pi k n / N}
    x[n]   = \\frac{1}{N} \\sum_{k=0}^{N-1} X[k]\\, e^{+j 2\\pi k n / N}

* forward transform: no scaling,
* inverse transform: divide by N.

Under these conventions Parseval's theorem for a length-N sequence reads

.. math::

    \\sum_n |x[n]|^2 = \\frac{1}{N} \\sum_k |X[k]|^2 .

The implementation uses a hand-written vectorized radix-2 Cooley-Tukey FFT
for power-of-two lengths (all selectable sizes in this tool are powers of
two) and a direct matrix DFT otherwise. This module is deliberately free of
numpy.fft calls so the arithmetic is genuinely implemented here and can be
audited on its own; the test-suite cross-checks it against numpy.fft.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _bit_reverse(x: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Reorder a length-N (N power of two) array into bit-reversed order."""
    n = x.shape[0]
    # Bit-reversed permutation via the classic doubling recurrence.
    perm = np.zeros(n, dtype=np.int64)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        perm[i] = j
    return x[perm]


def _fft_pow2(x: NDArray[np.complexfloating]) -> NDArray[np.complex128]:
    """In-place iterative radix-2 decimation-in-time FFT.

    `x` is treated with twiddle sign ``-1`` (forward). The inverse is obtained
    by conjugating the input/output (see :func:`idft`).
    """
    n = x.shape[0]
    a = _bit_reverse(np.asarray(x, dtype=np.complex128))

    length = 2
    while length <= n:
        half = length >> 1
        # All twiddle factors for this butterfly stage.
        angles = -2j * np.pi * np.arange(half) / length
        w = np.exp(angles)
        # Butterfly across every group simultaneously: reshape so columns are
        # the within-group index and let broadcasting handle the twiddles.
        a = a.reshape(n // length, length)
        even = a[:, :half]
        odd = w * a[:, half:]
        a = np.concatenate([even + odd, even - odd], axis=1).reshape(n)
        length <<= 1
    return a


def _dft_direct(x: NDArray[np.complexfloating]) -> NDArray[np.complex128]:
    """Brute-force O(N^2) DFT, used for non-power-of-two lengths."""
    n = x.shape[0]
    xc = np.asarray(x, dtype=np.complex128)
    idx = np.arange(n)
    # E[k, n] = exp(-j 2 pi k n / N)
    e = np.exp(-2j * np.pi * np.outer(idx, idx) / n)
    return e @ xc


def dft(x: NDArray[np.number]) -> NDArray[np.complex128]:
    """Forward DFT of a 1-D real or complex sequence."""
    xa = np.asarray(x)
    if xa.ndim != 1:
        raise ValueError("dft expects a 1-D sequence")
    n = xa.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.complex128)
    if n & (n - 1) == 0:
        return _fft_pow2(xa)
    return _dft_direct(xa)


def idft(x: NDArray[np.number]) -> NDArray[np.complex128]:
    """Inverse DFT (1/N normalization), using the conjugation trick."""
    xa = np.asarray(x)
    if xa.ndim != 1:
        raise ValueError("idft expects a 1-D sequence")
    n = xa.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.complex128)
    if n & (n - 1) == 0:
        return np.conj(_fft_pow2(np.conj(xa))) / n
    return np.conj(_dft_direct(np.conj(xa))) / n


def dft_frequencies(n: int, fs: float) -> NDArray[np.float64]:
    """Bin center frequencies in Hz for a length-N DFT at rate `fs`."""
    return np.arange(n, dtype=np.float64) * (fs / n)


def positive_frequencies(n: int, fs: float) -> NDArray[np.float64]:
    """Frequencies for the one-sided view 0 ... fs/2 (inclusive bins)."""
    k = n // 2 + 1
    return np.arange(k, dtype=np.float64) * (fs / n)


def magnitude_db(xk: NDArray[np.complexfloating], floor_db: float = -120.0) -> NDArray[np.float64]:
    """Magnitude spectrum in dB, floored at `floor_db` (handles zeros)."""
    mag = np.abs(xk)
    db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    return np.maximum(db, floor_db)


def analyze(
    signal: NDArray[np.number],
    window: NDArray[np.number] | None = None,
    n_padded: int | None = None,
) -> NDArray[np.complex128]:
    """Window (optionally), zero-pad (optionally), then forward DFT."""
    x = np.asarray(signal, dtype=np.float64)
    if window is not None:
        w = np.asarray(window, dtype=np.float64)
        if w.shape != x.shape:
            raise ValueError("window length must equal signal length")
        x = x * w
    if n_padded is not None and n_padded != x.shape[0]:
        x = np.pad(x, (0, n_padded - x.shape[0]))
    return dft(x)
