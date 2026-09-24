"""Short-time Fourier analysis: framing, windowed per-frame DFT, reconstruction.

This is the arithmetic core of the time-frequency ("spectrogram") feature.
It deliberately lives in its own module — it must hold up when tested
directly, not merely as a data source for the heatmap — but it reuses the
same transform kernel as the single-frame analysis (:func:`app.dft.analyze`),
so there is exactly one FFT implementation in the backend.

Framing convention (centered, the textbook STFT layout)
-------------------------------------------------------
The length-``M`` signal is regarded as zero outside the recording and
surrounded by ``L//2`` zeros. Analysis frames of length ``frame_length``
(``L``) advance by ``hop`` (``H``) samples, with frame ``j`` centered on
sample ``jH``:

    frame j covers original samples [jH - L//2, jH - L//2 + L - 1]
    num_frames = 1 + (M - 1) // H           (requires M >= L)

so the first column is stamped at ``t = 0`` and the columns keep arriving
while their centers lie inside the signal. Consequences worth knowing:

* a signal that exactly fills one frame (``M == L``) produces one column
  when ``hop == L`` — a legal, non-erroring degenerate result;
* with genuine overlap (``hop <= L/2``) every signal sample is seen by at
  least one frame at every window, including the boundaries, thanks to the
  zero surroundings;
* hop == L (no overlap) cannot cover the signal with centered frames and is
  rejected at reconstruction time rather than producing silent zeros.

Frequency rows are the usual bins ``k * fs / L`` Hz; only the one-sided
rows ``0 .. fs/2`` (``L/2 + 1``) are needed for the spectrogram, while the
inverse entry point consumes full length-``L`` spectra.

Reconstruction (weighted overlap-add)
-------------------------------------
Each framed spectrum is inverse-transformed; frames are overlap-added using
the analysis window itself as synthesis weight,

    y[n] = sum_j w[n - jH + L//2] * frame_j[...]  /
           sum_j w[n - jH + L//2]^2 .

On a covered sample the numerator is ``x[n] * sum_j w^2`` (windowing cancels
algebraically), so ``y[n] = x[n]`` to machine precision. Samples whose
squared-window coverage is exactly zero — the information is genuinely
absent from the frames, e.g. zero-edge windows placed without overlap — are
rejected with an explanatory :class:`~app.errors.BadRequest`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import ALLOWED_FRAME_LENGTHS, MAX_STFT_FRAMES
from .dft import analyze, idft
from .errors import BadRequest
from .windows import get_window

# A frame starts this many samples before its stamped time center.
# (All allowed frame lengths are even; the center is reported at jH, the
# conventional "frame centered on sample jH", matching librosa center=True.)


def validate_frame_params(frame_length: int, hop: int) -> None:
    """Check the product-level frame-length ladder and hop sanity.

    Raises :class:`BadRequest` (message names the offending value) for frame
    lengths outside the allowed档位, non-integer/non-positive hops, or hops
    larger than the frame length (which would skip samples).
    """
    if isinstance(frame_length, bool) or not isinstance(frame_length, int):
        raise BadRequest("frame_length must be an integer")
    if frame_length not in ALLOWED_FRAME_LENGTHS:
        allowed = ", ".join(str(v) for v in sorted(ALLOWED_FRAME_LENGTHS))
        raise BadRequest(
            f"frame_length must be one of {allowed}; got {frame_length}"
        )
    if isinstance(hop, bool) or not isinstance(hop, int):
        raise BadRequest("hop must be an integer number of samples")
    if hop <= 0:
        raise BadRequest(f"hop must be a positive integer; got {hop}")
    if hop > frame_length:
        raise BadRequest(
            f"hop ({hop}) must not exceed frame_length ({frame_length}); "
            "a hop larger than the frame would skip samples and leave gaps "
            "in the time-frequency picture"
        )


def num_frames(signal_length: int, frame_length: int, hop: int) -> int:
    """Number of centered analysis columns for a signal (requires M >= L)."""
    if signal_length < frame_length:
        return 0
    return 1 + (signal_length - 1) // hop


def _validate(signal_length: int, frame_length: int, hop: int) -> int:
    """Validate framing parameters against a concrete signal length."""
    validate_frame_params(frame_length, hop)
    if signal_length < frame_length:
        raise BadRequest(
            f"signal is too short to fill one frame: {signal_length} samples "
            f"for frame_length {frame_length}"
        )
    k = num_frames(signal_length, frame_length, hop)
    if k > MAX_STFT_FRAMES:
        raise BadRequest(
            f"frame count {k} exceeds the limit of {MAX_STFT_FRAMES}; "
            "increase the hop or shorten the signal"
        )
    return k


def frame_signal(
    signal: NDArray[np.number], frame_length: int, hop: int
) -> NDArray[np.float64]:
    """Slice a 1-D signal into centered, zero-surrounded analysis frames.

    Returns a ``(num_frames, frame_length)`` array (one frame per row);
    samples reaching before 0 or past the signal end are zero.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise BadRequest("signal must be a 1-D sequence")
    m = int(x.shape[0])
    k = _validate(m, frame_length, hop)
    half = frame_length // 2
    # L zeros of head room is plenty for the leftmost frame; L of tail room
    # covers the last (potentially overhanging) frame.
    padded = np.concatenate([np.zeros(frame_length), x, np.zeros(frame_length)])
    # Original index jH - half maps to padded index jH - half + frame_length.
    starts = frame_length - half + np.arange(k, dtype=np.int64) * hop
    idx = starts[:, None] + np.arange(frame_length)[None, :]
    return np.asarray(padded[idx], dtype=np.float64).copy()


def frame_times(
    signal_length: int, frame_length: int, hop: int, fs: float
) -> NDArray[np.float64]:
    """Time center (seconds) of every analysis column: j * hop / fs."""
    k = _validate(signal_length, frame_length, hop)
    return np.arange(k, dtype=np.float64) * hop / fs


def stft(
    signal: NDArray[np.number],
    fs: float,
    frame_length: int,
    hop: int,
    window: str = "hann",
    beta: float | None = None,
) -> dict[str, NDArray]:
    """Short-time Fourier transform of a real signal.

    Returns a dict with:

    * ``frames``     — complex ``(num_frames, frame_length)`` full spectra,
                       the expected input of :func:`istft`;
    * ``windowed``   — the windowed time-domain frames actually transformed;
    * ``times``      — column time centers in seconds;
    * ``frequencies``— bin frequencies in Hz (all ``frame_length`` rows);
    * ``window``     — the analysis window coefficients.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise BadRequest("signal must be a 1-D sequence")
    if x.size == 0:
        raise BadRequest("signal must contain at least one sample")
    if not np.all(np.isfinite(x)):
        raise BadRequest("signal contains non-finite values (NaN/Infinity)")
    if isinstance(fs, bool) or not isinstance(fs, (int, float)) or not np.isfinite(fs):
        raise BadRequest("sampling rate must be a finite number")
    if fs <= 0:
        raise BadRequest("sampling rate must be positive (> 0)")

    k = _validate(int(x.shape[0]), frame_length, hop)
    w = get_window(window, frame_length, beta)
    framed = frame_signal(x, frame_length, hop)  # (K, L), edges zero
    spectra = np.empty((k, frame_length), dtype=np.complex128)
    for j in range(k):
        # Reuse the single-frame kernel: window + DFT, no zero padding.
        spectra[j] = analyze(framed[j], window=w)

    return {
        "frames": spectra,
        "windowed": framed * w[np.newaxis, :],
        "times": frame_times(int(x.shape[0]), frame_length, hop, float(fs)),
        "frequencies": np.arange(frame_length, dtype=np.float64)
        * (float(fs) / frame_length),
        "window": w,
    }


def spectrogram(
    signal: NDArray[np.number],
    fs: float,
    frame_length: int,
    hop: int,
    window: str = "hann",
    beta: float | None = None,
) -> dict[str, NDArray]:
    """STFT plus the one-sided (frequency, time) intensity arrays.

    Matrix orientation matches the heatmap literally: rows index frequency
    (row 0 = DC), columns index frame time. Keys: ``power`` (|X|^2),
    ``magnitude`` (|X|), ``magnitude_db`` (floored at -120 dB), the
    ``times``/``frequencies`` axes, and framing metadata.
    """
    result = stft(signal, fs, frame_length, hop, window, beta)
    spectra = result["frames"]
    half = frame_length // 2 + 1
    one_sided = spectra[:, :half]  # (K, F)
    power = one_sided.real**2 + one_sided.imag**2
    magnitude = np.abs(one_sided)
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    magnitude_db = np.maximum(magnitude_db, -120.0)
    return {
        "power": power.T,  # (F, K)
        "magnitude": magnitude.T,
        "magnitude_db": magnitude_db.T,
        "times": result["times"],
        "frequencies": result["frequencies"][:half],
        "num_frames": int(spectra.shape[0]),
        "frame_length": int(frame_length),
        "hop": int(hop),
        "window": result["window"],
    }


def istft(
    frames: NDArray[np.complexfloating],
    frame_length: int,
    hop: int,
    signal_length: int,
    window: str = "hann",
    beta: float | None = None,
) -> NDArray[np.float64]:
    """Inverse STFT by weighted overlap-add.

    ``frames`` has shape ``(num_frames, frame_length)``: one *full*
    length-``frame_length`` spectrum per row, in the convention produced by
    :func:`stft`. ``signal_length`` is mandatory — it pins the expected
    output length, so a frame structure that does not line up with the
    forward analysis is rejected explicitly rather than silently yielding a
    longer or shorter signal.

    Raises :class:`BadRequest` for malformed frame arrays, frame counts that
    do not match ``(signal_length, frame_length, hop)``, or window/hop
    combinations whose squared-window coverage is zero on some sample (that
    sample's information is absent from the frames — add overlap).
    """
    z = np.asarray(frames)
    if z.ndim != 2:
        raise BadRequest(
            "frames must be a 2-D array with one spectrum per row "
            f"(got a {z.ndim}-D array)"
        )
    if not np.all(np.isfinite(z.real)) or not np.all(np.isfinite(z.imag)):
        raise BadRequest("frames contain non-finite values")
    validate_frame_params(frame_length, hop)
    if z.shape[1] != frame_length:
        raise BadRequest(
            f"frame spectrum length ({z.shape[1]}) does not match "
            f"frame_length ({frame_length})"
        )
    if isinstance(signal_length, bool) or not isinstance(signal_length, int):
        raise BadRequest("signal_length must be an integer")
    if signal_length < frame_length:
        raise BadRequest(
            f"signal_length ({signal_length}) is shorter than frame_length "
            f"({frame_length}); no matching forward analysis exists"
        )

    expected = num_frames(signal_length, frame_length, hop)
    if z.shape[0] != expected:
        raise BadRequest(
            f"frame count mismatch: {z.shape[0]} spectra supplied but "
            f"frame_length={frame_length}, hop={hop}, signal_length="
            f"{signal_length} require exactly {expected} frames"
        )

    w = get_window(window, frame_length, beta)
    half = frame_length // 2
    acc = np.zeros(signal_length, dtype=np.float64)
    weight = np.zeros(signal_length, dtype=np.float64)
    for j in range(expected):
        frame_t = np.real(idft(z[j]))  # real spectra -> real time frame
        start = j * hop - half  # original sample at frame position 0
        lo, hi = max(0, start), min(signal_length, start + frame_length)
        if hi <= lo:
            continue
        p0, p1 = lo - start, hi - start
        acc[lo:hi] += w[p0:p1] * frame_t[p0:p1]
        weight[lo:hi] += w[p0:p1] ** 2

    missing = weight <= 1e-12
    if np.any(missing):
        first = int(np.argmax(missing))
        raise BadRequest(
            f"cannot reconstruct sample n={first}: the squared analysis "
            f"window {window!r} with hop {hop} (frame_length "
            f"{frame_length}) has zero coverage there, so that sample "
            "carries no information in the frames. Use genuine overlap "
            "(hop <= frame_length/2)"
        )
    return acc / weight
