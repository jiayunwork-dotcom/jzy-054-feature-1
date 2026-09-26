"""Service layer: request validation + orchestration over the math modules.

The math kernel (:mod:`app.dft`, :mod:`app.windows`, :mod:`app.filtering`,
:mod:`app.aliasing`, :mod:`app.stft`) stays free of API concerns; this module
validates the product-level rules (allowed sizes, positive rates, ...) and
converts numpy arrays to plain Python lists for the Pydantic responses.
"""

from __future__ import annotations

import numpy as np

from . import aliasing, filtering, stft
from .config import ALLOWED_N, ALLOWED_PADDED_N
from .dft import analyze, dft_frequencies, idft, magnitude_db
from .errors import BadRequest
from .windows import get_window, window_metrics


def _finite_real_signal(signal: list[float]) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise BadRequest("signal must be a 1-D list of numbers")
    if arr.size == 0:
        raise BadRequest("signal must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise BadRequest("signal contains non-finite values (NaN/Infinity)")
    return arr


def _check_fs(fs: float) -> None:
    if not isinstance(fs, (int, float)) or not np.isfinite(fs):
        raise BadRequest("sampling rate must be a finite number")
    if fs <= 0:
        raise BadRequest("sampling rate must be positive (> 0)")


def _check_n(n: int) -> None:
    if n not in ALLOWED_N:
        allowed = ", ".join(str(v) for v in sorted(ALLOWED_N))
        raise BadRequest(f"N must be one of {allowed}; got {n}")


def _check_windows(names_betas: list[tuple[str, float | None]]) -> None:
    if not names_betas:
        raise BadRequest("at least one window must be requested")
    seen: set[tuple[str, float | None]] = set()
    for name, beta in names_betas:
        # get_window itself raises BadRequest on unknown names / missing beta.
        get_window(name, 8, beta)
        key = (name, None if beta is None else round(float(beta), 6))
        if key in seen:
            raise BadRequest(f"window {name!r} with the same beta requested twice")
        seen.add(key)


def dft_service(req) -> dict:
    """Validate a /dft request and compute spectra for every requested window."""
    x = _finite_real_signal(req.signal)
    _check_fs(req.fs)
    _check_n(req.n)

    m = x.shape[0]
    if m > req.n:
        raise BadRequest(
            f"signal length ({m}) exceeds the selected N ({req.n}); "
            "increase N or build a shorter signal"
        )

    n_padded = req.n
    if req.padded_n is not None:
        if req.padded_n not in ALLOWED_PADDED_N:
            allowed = ", ".join(str(v) for v in sorted(ALLOWED_PADDED_N))
            raise BadRequest(f"padded N must be one of {allowed}")
        if req.padded_n < m:
            raise BadRequest(
                f"padded length ({req.padded_n}) is shorter than the signal "
                f"({m}); zero-padding can only extend the signal"
            )
        n_padded = req.padded_n

    window_specs = [(w.name, w.beta) for w in req.windows]
    _check_windows(window_specs)

    freqs = dft_frequencies(n_padded, float(req.fs))

    spectra = []
    for name, beta in window_specs:
        w = get_window(name, m, beta)
        xk = analyze(x, window=w, n_padded=n_padded)
        spectra.append(
            {
                "window": name,
                "beta": None if beta is None else float(beta),
                "real": [float(v) for v in xk.real],
                "imag": [float(v) for v in xk.imag],
                "magnitude": [float(v) for v in np.abs(xk)],
                "phase": [float(v) for v in np.angle(xk)],
                "power": [float(v) for v in (xk.real**2 + xk.imag**2)],
                "magnitude_db": [float(v) for v in magnitude_db(xk)],
            }
        )

    return {
        "n": int(req.n),
        "n_padded": int(n_padded),
        "signal_length": int(m),
        "fs": float(req.fs),
        "frequencies": [float(f) for f in freqs],
        "spectra": spectra,
    }


def idft_service(req) -> list[float]:
    real = np.asarray(req.real, dtype=np.float64)
    if real.ndim != 1 or real.size == 0:
        raise BadRequest("real part must be a non-empty 1-D list")
    if req.imag is None:
        imag = np.zeros_like(real)
    else:
        imag = np.asarray(req.imag, dtype=np.float64)
    if imag.shape != real.shape:
        raise BadRequest("real and imaginary parts must have equal length")
    if not (np.all(np.isfinite(real)) and np.all(np.isfinite(imag))):
        raise BadRequest("spectrum contains non-finite values")
    x = idft(real + 1j * imag)
    return [float(v) for v in np.real_if_close(x, tol=1000)]


def windows_service(req) -> list[dict]:
    if req.n < 1:
        raise BadRequest("window length must be a positive integer")
    out = []
    for spec in req.windows:
        w = get_window(spec.name, req.n, spec.beta)
        metrics = window_metrics(spec.name, spec.beta)
        out.append(
            {
                "name": spec.name,
                "beta": None if spec.beta is None else float(spec.beta),
                "coefficients": [float(v) for v in w],
                "mainlobe_bins": metrics["mainlobe_bins"],
                "peak_sidelobe_db": metrics["peak_sidelobe_db"],
            }
        )
    return out


def filter_service(req) -> dict:
    x = _finite_real_signal(req.signal)
    _check_fs(req.fs)
    result = filtering.apply_filter(
        x,
        float(req.fs),
        req.mode,
        None if req.cutoff_low is None else float(req.cutoff_low),
        None if req.cutoff_high is None else float(req.cutoff_high),
    )
    n = x.shape[0]
    freqs = dft_frequencies(n, float(req.fs))
    xk = result["spectrum"]
    return {
        "filtered_signal": [float(v) for v in result["filtered"]],
        "spectrum_real": [float(v) for v in xk.real],
        "spectrum_imag": [float(v) for v in xk.imag],
        "mask": [int(v) for v in result["mask"]],
        "frequencies": [float(f) for f in freqs],
        "input_energy": result["input_energy"],
        "output_energy": result["output_energy"],
        "residual_high_energy": result["residual_high_energy"],
    }


def sampling_service(req) -> dict:
    result = aliasing.sampling_demo(
        float(req.signal_freq),
        float(req.fs),
        n_samples=int(req.n_samples),
        phase=float(req.phase),
    )
    return {
        "original_t": _to_list(result["original_t"]),
        "original_y": _to_list(result["original_y"]),
        "sample_t": _to_list(result["sample_t"]),
        "sample_y": _to_list(result["sample_y"]),
        "reconstructed_t": _to_list(result["reconstructed_t"]),
        "reconstructed_y": _to_list(result["reconstructed_y"]),
        "aliased": bool(result["aliased"]),
        "apparent_freq_hz": float(result["apparent_freq_hz"]),
        "nyquist_hz": float(result["nyquist_hz"]),
    }


def stft_service(req) -> dict:
    """Validate a /stft request and compute the spectrogram (+ axes)."""
    x = _finite_real_signal(req.signal)
    _check_fs(req.fs)
    if req.pad_left < 0:
        raise BadRequest("pad_left must be non-negative")
    # get_window raises BadRequest on unknown names / missing beta.
    get_window(req.window.name, 8, req.window.beta)

    result = stft.spectrogram(
        x,
        float(req.fs),
        req.frame_len,
        req.hop,
        req.window.name,
        req.window.beta,
        pad_left=int(req.pad_left),
    )
    out = {
        "frame_len": int(req.frame_len),
        "hop": int(req.hop),
        "fs": float(req.fs),
        "window": req.window.name,
        "beta": None if req.window.beta is None else float(req.window.beta),
        "num_frames": result["num_frames"],
        "num_bins": result["num_bins"],
        "times": [float(v) for v in result["times"]],
        "frequencies": [float(v) for v in result["frequencies"]],
        "magnitude": [[float(v) for v in row] for row in result["magnitude"]],
        "frame_duration_s": result["frame_duration_s"],
        "time_step_s": result["time_step_s"],
        "freq_step_hz": result["freq_step_hz"],
        "pad_left": int(req.pad_left),
        "signal_length": int(x.shape[0]),
    }
    if req.include_frames:
        spectra, _ = stft.stft_frames(
            x,
            req.frame_len,
            req.hop,
            req.window.name,
            req.window.beta,
            pad_left=int(req.pad_left),
        )
        out["frames_real"] = [[float(v) for v in row] for row in spectra.real]
        out["frames_imag"] = [[float(v) for v in row] for row in spectra.imag]
    return out


def istft_service(req) -> dict:
    """Validate an /istft request and reconstruct the time signal."""
    real = _frame_matrix(req.frames_real, "frames_real")
    if req.frames_imag is None:
        imag = np.zeros_like(real)
    else:
        imag = _frame_matrix(req.frames_imag, "frames_imag")
        if imag.shape != real.shape:
            raise BadRequest(
                "frame structure mismatch: frames_real has shape "
                f"{real.shape} but frames_imag has shape {imag.shape}"
            )
    if req.pad_left < 0:
        raise BadRequest("pad_left must be non-negative")
    if req.signal_length < 1:
        raise BadRequest("signal_length must be a positive integer")

    recon = stft.istft(
        real + 1j * imag,
        req.hop,
        req.window.name,
        req.window.beta,
        pad_left=int(req.pad_left),
        signal_length=int(req.signal_length),
    )
    return {
        "signal": [float(v) for v in recon],
        "num_frames": int(real.shape[0]),
        "frame_len": int(real.shape[1]),
    }


def _frame_matrix(rows, field: str) -> np.ndarray:
    """Parse a JSON nested list into a finite 2-D array (or explain why not)."""
    try:
        arr = np.asarray(rows, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise BadRequest(
            f"frame structure mismatch: {field} rows have inconsistent lengths"
        ) from exc
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise BadRequest(f"{field} must be a non-empty 2-D matrix (frames x bins)")
    if arr.shape[1] == 0:
        raise BadRequest(f"{field} rows must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise BadRequest(f"{field} contains non-finite values (NaN/Infinity)")
    return arr


def _to_list(value):
    if isinstance(value, np.ndarray):
        return [float(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value
