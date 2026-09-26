"""Short-time Fourier analysis: framing, per-frame windowed transforms, and
weighted overlap-add reconstruction.

This module is the arithmetic backbone of the time-frequency (spectrogram)
view. It deliberately lives apart from the single-shot transform in
:mod:`app.dft`, but *reuses* that transform kernel and the shared window
definitions — there is exactly one FFT and one window table in this backend.

Framing model
-------------
The signal is cut into frames of ``frame_len`` samples starting every
``hop`` samples::

    frame m  =  x[m*hop : m*hop + frame_len]          m = 0 .. F-1

With no padding (the spectrogram view) only complete frames are produced
and a signal of exactly ``frame_len`` samples yields exactly one frame;
shorter signals are rejected. For the round trip the signal is first
extended with ``pad_left`` zeros on both sides (the right side is further
extended so the frame starts land exactly on the hop grid and every padded
sample is covered).

Round-trip identity
-------------------
Analysis multiplies each frame by the window ``w`` before the DFT. Synthesis
(:func:`istft`) inverse-transforms, multiplies by ``w`` again, overlap-adds,
and divides sample-wise by the window energy envelope ``sum_m w^2`` shifted
over each position. Wherever that envelope is strictly positive the
reconstruction is *exact* (up to floating-point error) — this is the
weighted-overlap-add (WOLA) identity the test-suite guards. With zero-padded
edges (``pad_left = frame_len // 2``) and reasonable overlap (hop <=
frame_len/2 for the tapered windows) every sample of the original signal has
positive envelope; settings that violate this (e.g. a Hann window with
``hop == frame_len``, where window zeros coincide with uncovered samples)
are rejected with an explanatory error instead of silently returning
garbage.
"""

from __future__ import annotations

import numpy as np

from .config import ALLOWED_FRAME_N
from .dft import dft, idft, positive_frequencies
from .errors import BadRequest
from .windows import get_window

# Envelope floor below which a sample is considered unrecoverable.
_ENVELOPE_EPS = 1e-10


def check_frame_settings(frame_len: int, hop: int) -> None:
    """Product rules shared by the forward and inverse paths.

    * ``frame_len`` must be one of the selectable gears (ALLOWED_FRAME_N);
    * ``hop`` must be a positive integer not exceeding ``frame_len`` — a
      larger hop would leave samples unobserved between frames and the
      spectrogram would lie by omission.
    """
    if not isinstance(frame_len, (int, np.integer)) or isinstance(frame_len, bool):
        raise BadRequest("frame length must be an integer")
    if frame_len not in ALLOWED_FRAME_N:
        allowed = ", ".join(str(v) for v in sorted(ALLOWED_FRAME_N))
        raise BadRequest(f"frame length must be one of {allowed}; got {frame_len}")
    if not isinstance(hop, (int, np.integer)) or isinstance(hop, bool):
        raise BadRequest("hop size must be an integer number of samples")
    if hop < 1:
        raise BadRequest("hop size must be positive (>= 1 sample)")
    if hop > frame_len:
        raise BadRequest(
            f"hop size ({hop}) must not exceed the frame length ({frame_len}); "
            "otherwise samples between frames would be skipped"
        )


def frame_signal(
    x: np.ndarray,
    frame_len: int,
    hop: int,
    pad_left: int = 0,
    pad_right: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut ``x`` into overlapping frames; return ``(frames, starts)``.

    ``frames`` has shape ``(num_frames, frame_len)``; ``starts[m]`` is the
    index of frame ``m`` within the (possibly zero-padded) signal.

    Without padding (the spectrogram view) only complete frames of the raw
    signal are produced, so a signal of exactly ``frame_len`` samples yields
    exactly one frame. With ``pad_left``/``pad_right`` (the round-trip path)
    the signal is extended by zeros on both sides — default
    ``pad_right = pad_left`` — plus up to ``hop - 1`` extra zeros on the
    right so the last frame ends exactly on the hop grid and every padded
    sample is covered.
    """
    check_frame_settings(frame_len, hop)
    if pad_left < 0:
        raise BadRequest("pad_left must be non-negative")
    if pad_right is None:
        pad_right = pad_left
    if pad_right < 0:
        raise BadRequest("pad_right must be non-negative")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise BadRequest("signal must be a 1-D sequence")
    n = x.shape[0]
    total = n + pad_left + pad_right
    if total < frame_len:
        raise BadRequest(
            f"signal length ({n}) is shorter than one frame ({frame_len}); "
            "cannot form even a single segment"
        )
    if pad_left + pad_right > 0:
        # Extend the right edge so (total - frame_len) is a multiple of hop.
        remainder = (total - frame_len) % hop
        if remainder:
            pad_right += hop - remainder
    xp = np.pad(x, (pad_left, pad_right))
    num_frames = 1 + (xp.shape[0] - frame_len) // hop
    starts = np.arange(num_frames, dtype=np.int64) * hop
    # Strided view: row m is xp[starts[m] : starts[m] + frame_len].
    frames = np.lib.stride_tricks.as_strided(
        xp,
        shape=(num_frames, frame_len),
        strides=(xp.strides[0] * hop, xp.strides[0]),
    ).copy()
    return frames, starts


def stft_frames(
    x: np.ndarray,
    frame_len: int,
    hop: int,
    window_name: str,
    beta: float | None = None,
    pad_left: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed per-frame DFT.

    Returns ``(spectra, starts)`` where ``spectra[m]`` is the length-
    ``frame_len`` complex DFT of frame ``m`` multiplied by the window.
    """
    frames, starts = frame_signal(x, frame_len, hop, pad_left)
    w = get_window(window_name, frame_len, beta)
    spectra = np.stack([dft(frame * w) for frame in frames])
    return spectra, starts


def spectrogram(
    x: np.ndarray,
    fs: float,
    frame_len: int,
    hop: int,
    window_name: str,
    beta: float | None = None,
    pad_left: int = 0,
) -> dict:
    """One-sided amplitude spectrogram plus its two physical axes.

    Returns a dict with

    * ``magnitude``  — ``(num_frames, frame_len//2 + 1)`` matrix, normalized
      by the window's coherent gain so a steady sine of amplitude ``A``
      reads ``A`` at its peak row (DC/Nyquist rows are not doubled);
    * ``times``      — center time of each frame in seconds;
    * ``frequencies``— bin center frequency of each row in Hz (0 … fs/2);
    * the resolution figures the UI prints: ``frame_duration_s`` (how wide a
      column is in time), ``time_step_s`` (hop in seconds) and
      ``freq_step_hz`` (how tall a row is in frequency).
    """
    spectra, starts = stft_frames(x, frame_len, hop, window_name, beta, pad_left)
    w = get_window(window_name, frame_len, beta)
    coherent_gain = float(np.sum(w))

    half = frame_len // 2 + 1
    mag = np.abs(spectra[:, :half]) / coherent_gain
    # One-sided amplitude: interior bins hold half the energy of a real
    # component (the mirror half lives at negative frequencies).
    mag[:, 1 : half - 1] *= 2.0

    times = (starts + frame_len / 2.0 - pad_left) / float(fs)
    freqs = positive_frequencies(frame_len, float(fs))
    return {
        "magnitude": mag,
        "times": times,
        "frequencies": freqs,
        "num_frames": int(mag.shape[0]),
        "num_bins": int(half),
        "frame_duration_s": frame_len / float(fs),
        "time_step_s": hop / float(fs),
        "freq_step_hz": float(fs) / frame_len,
    }


def istft(
    spectra: np.ndarray,
    hop: int,
    window_name: str,
    beta: float | None = None,
    pad_left: int = 0,
    signal_length: int | None = None,
) -> np.ndarray:
    """Inverse short-time Fourier transform (weighted overlap-add).

    ``spectra`` is the ``(num_frames, frame_len)`` complex matrix produced by
    :func:`stft_frames`. Each frame is inverse-transformed, multiplied by the
    same window, overlap-added, and normalized by the window energy envelope;
    the ``pad_left`` boundary extension is stripped and exactly
    ``signal_length`` samples are returned.

    Raises :class:`BadRequest` when the frame structure cannot correspond to
    the claimed signal (frames cannot cover ``signal_length`` samples) or
    when the window/hop combination leaves some samples with zero envelope
    weight (no exact reconstruction possible — more overlap is needed).
    """
    X = np.asarray(spectra, dtype=np.complex128)
    if X.ndim != 2 or X.shape[0] == 0:
        raise BadRequest("frames must be a non-empty 2-D matrix (frames x bins)")
    num_frames, frame_len = X.shape
    check_frame_settings(frame_len, hop)
    if pad_left < 0:
        raise BadRequest("pad_left must be non-negative")
    if signal_length is None:
        signal_length = (num_frames - 1) * hop + frame_len - 2 * pad_left
    if signal_length < 1:
        raise BadRequest("signal_length must be a positive integer")

    covered = (num_frames - 1) * hop + frame_len
    if pad_left + signal_length > covered:
        raise BadRequest(
            f"frame structure inconsistent with signal_length: {num_frames} "
            f"frames of {frame_len} samples at hop {hop} cover {covered} "
            f"samples, but pad_left ({pad_left}) + signal_length "
            f"({signal_length}) needs {pad_left + signal_length}"
        )

    w = get_window(window_name, frame_len, beta)
    out = np.zeros(covered, dtype=np.float64)
    envelope = np.zeros(covered, dtype=np.float64)
    w2 = w * w
    for m in range(num_frames):
        s = m * hop
        # The analyzed signal is real, so the inverse frame is real up to
        # numerical noise; take the real part explicitly.
        out[s : s + frame_len] += w * idft(X[m]).real
        envelope[s : s + frame_len] += w2

    lo = pad_left
    hi = pad_left + signal_length
    segment_envelope = envelope[lo:hi]
    if np.any(segment_envelope < _ENVELOPE_EPS):
        raise BadRequest(
            "this window/hop combination leaves some samples with zero "
            "reconstruction weight (the window is zero wherever no "
            "overlapping frame covers them); increase the overlap "
            "(smaller hop) or pick another window"
        )
    return out[lo:hi] / segment_envelope
