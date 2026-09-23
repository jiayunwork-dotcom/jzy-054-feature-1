"""Window functions and their spectral figures of merit.

All windows are generated in the *periodic* convention
(``w[n]`` for ``n = 0 .. N-1`` with ``N`` in the trig argument), matching
the common definition used for spectral analysis and numpy's default.

Supported names: ``rect``, ``hann``, ``hamming``, ``blackman``, ``kaiser``.
The Kaiser window requires a beta parameter.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

import numpy as np

from .config import WINDOW_NAMES, WINDOW_TABLE
from .dft import dft
from .errors import BadRequest

T = TypeVar("T", float, np.ndarray)


def _i0(x: T) -> T:
    """Modified Bessel function of the first kind, order zero.

    Series expansion ``I0(x) = sum_{k=0}^inf ((x/2)^k / k!)^2``.
    Implemented here so the backend only needs numpy. Works for floats and
    arrays alike.
    """
    arr = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(arr)
    term = np.ones_like(arr)
    half = arr / 2.0
    # Each subsequent term is term * (half/(k+1))**2; stop at machine precision.
    for k in range(100):
        result = result + term
        term = term * (half / (k + 1)) ** 2
        if np.all(np.abs(term) < 1e-18 * np.maximum(np.abs(result), 1.0)):
            break
    if np.isscalar(x) and getattr(x, "ndim", 0) == 0:
        return float(result)
    return result


def get_window(name: str, n: int, beta: float | None = None) -> np.ndarray:
    """Return a length-n window.

    Raises :class:`BadRequest` for unknown names or a Kaiser window without
    a finite beta.
    """
    if n < 1:
        raise BadRequest("window length must be a positive integer")
    if name not in WINDOW_NAMES:
        raise BadRequest(
            f"unknown window {name!r}; expected one of {', '.join(WINDOW_NAMES)}"
        )

    if name == "kaiser":
        if beta is None:
            raise BadRequest("kaiser window requires a beta parameter")
        try:
            beta_val = float(beta)
        except (TypeError, ValueError) as exc:
            raise BadRequest("kaiser beta must be a number") from exc
        if not np.isfinite(beta_val):
            raise BadRequest("kaiser beta must be a finite number")
        if beta_val < 0:
            raise BadRequest("kaiser beta must be non-negative")
        k = np.arange(n, dtype=np.float64)
        arg = 2.0 * k / n - 1.0  # -1 .. 1 (excludes the duplicate endpoint)
        inner = np.maximum(0.0, 1.0 - arg * arg)
        w = _i0(beta_val * np.sqrt(inner)) / _i0(beta_val)
        return np.asarray(w, dtype=np.float64)

    k = np.arange(n, dtype=np.float64)
    if name == "rect":
        return np.ones(n, dtype=np.float64)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * k / n)
    # blackman
    return (
        0.42
        - 0.5 * np.cos(2.0 * np.pi * k / n)
        + 0.08 * np.cos(4.0 * np.pi * k / n)
    )


@lru_cache(maxsize=64)
def _measured_metrics(name: str, beta_key: float | None) -> tuple[float, float]:
    """Numerically measure main-lobe width (bins) and peak sidelobe (dB).

    A fixed length-64 window is zero-padded to 8192 samples and transformed;
    the dense grid resolves the window's discrete-time Fourier transform well
    enough to locate the first null and the highest sidelobe peak.
    """
    n = 64
    dense = 8192
    w = get_window(name, n, beta_key)
    spectrum = dft(np.pad(w, (0, dense - n)))
    mag = np.abs(spectrum)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-15) / mag[0])

    # Frequency coordinate in "DFT bins of an N-point transform":
    # dense grid index / n corresponds to k of the length-N DFT.
    coord = np.arange(dense) * (dense / n) / dense  # = index / n

    # First null: first local minimum of the magnitude after bin 0.
    # Search the linear (not dB) magnitude for a change from descending to
    # ascending between neighboring dense-grid points.
    found_null = None
    i = 2
    while i < dense // 2:
        if mag[i - 1] > mag[i] and mag[i] <= mag[i + 1] and mag[i] < 0.5 * mag[0]:
            found_null = i
            break
        i += 1
    if found_null is None:  # pragma: no cover - all standard windows null
        mainlobe_bins = 2.0
    else:
        # Null-to-null width: -null..+null, linear interp the minimum location.
        x0, x1, x2 = coord[found_null - 1], coord[found_null], coord[found_null + 1]
        y0, y1, y2 = mag[found_null - 1], mag[found_null], mag[found_null + 1]
        # Quadratic interpolation of the minimum in grid steps.
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
        null_bin = (found_null + delta) / n
        mainlobe_bins = 2.0 * null_bin

    # Peak sidelobe: highest level after the first null.
    if found_null is not None:
        tail = mag_db[found_null + 1 : dense // 2]
        peak_sidelobe = float(np.max(tail))
    else:  # pragma: no cover
        peak_sidelobe = float(np.min(mag_db[1 : dense // 2]))
    return round(mainlobe_bins, 2), round(peak_sidelobe, 1)


def window_metrics(name: str, beta: float | None = None) -> dict[str, float | None]:
    """Main-lobe width in bins and peak sidelobe level in dB.

    The four classic windows return the standard design-table values from
    :data:`app.config.WINDOW_TABLE`; the Kaiser window (whose figures depend
    on beta) is measured numerically.
    """
    if name not in WINDOW_NAMES:
        raise BadRequest(
            f"unknown window {name!r}; expected one of {', '.join(WINDOW_NAMES)}"
        )
    if name == "kaiser":
        if beta is None:
            raise BadRequest("kaiser window requires a beta parameter")
        beta = float(beta)
        if not np.isfinite(beta) or beta < 0:
            raise BadRequest("kaiser beta must be a finite non-negative number")
        mainlobe, sidelobe = _measured_metrics("kaiser", round(beta, 4))
        return {"mainlobe_bins": mainlobe, "peak_sidelobe_db": sidelobe}
    table = dict(WINDOW_TABLE[name])
    return table
