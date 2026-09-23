"""Sampling theorem and aliasing analysis.

Decision rule (the Shannon-Nyquist criterion, asserted explicitly rather
than left to the drawing):

* a tone of frequency ``f`` sampled at rate ``fs`` is **aliased** when
  ``f > fs / 2``;
* the apparent ("reconstructed") frequency is then the folded frequency

  .. math:: f_{app} = |(f + f_s/2) \\bmod f_s - f_s/2|,

  which always lies in [0, fs/2].
"""

from __future__ import annotations

import numpy as np

from .errors import BadRequest


def apparent_frequency(f: float, fs: float) -> float:
    """Frequency in [0, fs/2] that a sampled tone is indistinguishable from."""
    return float(abs((f + fs / 2.0) % fs - fs / 2.0))


def is_aliased(signal_freq: float, fs: float) -> bool:
    """True exactly when the tone exceeds the Nyquist rate fs/2."""
    return float(signal_freq) > fs / 2.0


def _sinc_reconstruct(samples: np.ndarray, fs: float, t: np.ndarray) -> np.ndarray:
    """Ideal band-limited (sinc) reconstruction of `samples` at times `t`.

    ``x(t) = sum_n x[n] sinc(pi (t - nT)/T)`` with T = 1/fs.
    Vectorized; the diagonal ``t == nT`` cases are handled explicitly to
    avoid 0/0.
    """
    n = np.arange(samples.shape[0])
    # (t - nT)/T with T = 1/fs. Parenthesize carefully: fs multiplies the
    # whole difference, not n (that would compute n*fs instead of n/fs).
    dt = (t[:, None] - n[None, :] / fs) * fs
    # Evaluate the sinc only away from the removable singularity at dt == 0,
    # where its value is 1. (np.where evaluates both branches, so guard the
    # denominator explicitly instead of dividing through zero.)
    denom = np.pi * dt
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.divide(
            np.sin(denom),
            denom,
            out=np.ones_like(denom),
            where=denom != 0.0,
        )
    return ratio @ samples


def sampling_demo(
    signal_freq: float,
    fs: float,
    n_samples: int = 24,
    samples_per_interval: int = 30,
    phase: float = 0.0,
) -> dict:
    """Build all three traces for the sampling-theorem visualization.

    Returns:

    * ``original_t`` / ``original_y``: a densely sampled version of the true
      cosine (used as the reference "ground truth" curve),
    * ``sample_t`` / ``sample_y``: the actual discrete samples at rate fs,
    * ``reconstructed_t`` / ``reconstructed_y``: sinc interpolation of the
      samples — when aliased this follows the folded apparent tone,
    * the boolean ``aliased`` and the computed ``apparent_freq_hz``.
    """
    if not np.isfinite(signal_freq) or signal_freq < 0:
        raise BadRequest("signal frequency must be a non-negative number")
    if not np.isfinite(fs) or fs <= 0:
        raise BadRequest("sampling rate must be positive")
    if n_samples < 4:
        raise BadRequest("need at least 4 samples for reconstruction")

    duration = n_samples / fs
    # Dense grids covering [0, (n_samples-1)/T] so reconstruction at the
    # endpoints matches a sample rather than extrapolating beyond the data.
    end_t = (n_samples - 1) / fs
    sample_t = np.arange(n_samples) / fs
    dense_n = (n_samples - 1) * samples_per_interval + 1
    original_t = np.linspace(0.0, end_t, dense_n)

    original_y = np.cos(2.0 * np.pi * signal_freq * original_t + phase)
    sample_y = np.cos(2.0 * np.pi * signal_freq * sample_t + phase)
    reconstructed_y = _sinc_reconstruct(sample_y, fs, original_t)

    aliased = is_aliased(signal_freq, fs)
    f_app = apparent_frequency(signal_freq, fs)

    return {
        "original_t": original_t,
        "original_y": original_y,
        "sample_t": sample_t,
        "sample_y": sample_y,
        "reconstructed_t": original_t,
        "reconstructed_y": reconstructed_y,
        "aliased": aliased,
        "apparent_freq_hz": f_app,
        "nyquist_hz": fs / 2.0,
    }
