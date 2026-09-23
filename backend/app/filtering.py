"""Frequency-domain filtering.

The filter band is specified in physical frequency (Hz) on [0, fs/2].
Because the input is real, its DFT is conjugate-symmetric: when a positive
bin is kept/dropped its mirror bin ``N-k`` must be treated identically so
the inverse transform stays real.

Modes:

* ``lowpass``  keep 0 <= f <= cutoff_high (cutoff_low ignored / must equal it)
* ``highpass`` keep cutoff_low <= f <= fs/2
* ``bandpass`` keep cutoff_low <= f <= cutoff_high
"""

from __future__ import annotations

import numpy as np

from .dft import dft, idft
from .errors import BadRequest

FILTER_MODES = ("lowpass", "highpass", "bandpass")


def _validate_band(
    mode: str,
    fs: float,
    cutoff_low: float | None,
    cutoff_high: float | None,
) -> None:
    if mode not in FILTER_MODES:
        raise BadRequest(
            f"unknown filter mode {mode!r}; expected one of {', '.join(FILTER_MODES)}"
        )
    if not np.isfinite(fs) or fs <= 0:
        raise BadRequest("sampling rate must be positive")

    nyquist = fs / 2.0
    if mode == "lowpass":
        if cutoff_high is None:
            raise BadRequest("lowpass filter requires cutoff_high")
        if not np.isfinite(cutoff_high) or cutoff_high < 0 or cutoff_high > nyquist:
            raise BadRequest(
                f"cutoff must lie within [0, {nyquist:g}] Hz (the Nyquist band)"
            )
        return

    if mode == "highpass":
        if cutoff_low is None:
            raise BadRequest("highpass filter requires cutoff_low")
        if not np.isfinite(cutoff_low) or cutoff_low < 0 or cutoff_low > nyquist:
            raise BadRequest(
                f"cutoff must lie within [0, {nyquist:g}] Hz (the Nyquist band)"
            )
        return

    # bandpass
    if cutoff_low is None or cutoff_high is None:
        raise BadRequest("bandpass filter requires cutoff_low and cutoff_high")
    lo, hi = cutoff_low, cutoff_high
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise BadRequest("cutoff frequencies must be finite numbers")
    if lo < 0 or hi > nyquist:
        raise BadRequest(
            f"filter band [{lo:g}, {hi:g}] Hz is out of range "
            f"[0, {nyquist:g}] Hz"
        )
    if hi <= lo:
        raise BadRequest(
            f"cutoff_high ({hi:g} Hz) must be strictly greater than "
            f"cutoff_low ({lo:g} Hz)"
        )


def apply_filter(
    signal: np.ndarray,
    fs: float,
    mode: str,
    cutoff_low: float | None = None,
    cutoff_high: float | None = None,
) -> dict:
    """Zero out DFT bins outside the selected band, then inverse transform.

    Returns the filtered time signal, the masked spectrum and a small energy
    report (time-domain energy before/after, and the fraction of DFT energy
    left above the low cutoff — used by the tests and the UI).
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise BadRequest("signal must be a 1-D sequence")
    _validate_band(mode, fs, cutoff_low, cutoff_high)

    n = x.shape[0]
    xk = dft(x)
    bin_f = np.arange(n, dtype=np.float64) * (fs / n)
    # Folded frequencies 0..fs/2..0 (mirrored bins map back to the base band).
    folded_f = np.minimum(bin_f, fs - bin_f)

    if mode == "lowpass":
        keep = folded_f <= cutoff_high
    elif mode == "highpass":
        keep = folded_f >= cutoff_low
    else:
        keep = (folded_f >= cutoff_low) & (folded_f <= cutoff_high)

    xk_filtered = np.where(keep, xk, 0.0)
    y = idft(xk_filtered)
    # Symmetry of the mask keeps the imaginary part at numerical-noise level;
    # return the real reconstruction explicitly.
    y = np.real_if_close(y, tol=1000)
    y = np.asarray(y, dtype=np.float64)

    input_energy = float(np.sum(x * x))
    output_energy = float(np.sum(y * y))

    # Energy residing at frequencies above the low cutoff (spectral domain,
    # Parseval: sum|X|^2 / N). Helps prove "high frequency energy is gone".
    if mode == "bandpass":
        above_lo = folded_f > cutoff_high
    else:
        above_lo = folded_f > (cutoff_high if mode == "lowpass" else cutoff_low)
    residual_hi_energy = float(np.sum(np.abs(xk_filtered[above_lo]) ** 2) / n)

    return {
        "filtered": y,
        "spectrum": xk_filtered,
        "mask": keep.astype(np.int8),
        "input_energy": input_energy,
        "output_energy": output_energy,
        "residual_high_energy": residual_hi_energy,
    }
