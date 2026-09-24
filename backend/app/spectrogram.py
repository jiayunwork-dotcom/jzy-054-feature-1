"""Short-time Fourier analysis: overlapping windowed frames, STFT and ISTFT.

Unlike :mod:`app.dft`, which treats a whole segment as a single object, this
module slices a signal into a chain of overlapping frames, windows each frame
and transforms it independently; stacking the per-frame spectra along a time
axis yields a time-frequency (spectrogram) matrix

    time  (columns)  ->
    frequency (rows) v

Framing
-------
Frames of ``frame_length`` samples start at positions ``m * hop_length``.
Interior frames are those lying wholly inside the signal, i.e. with start
``s`` in ``0 .. L - frame_length``. Frames beyond the last interior frame
that still overlap the signal are appended as **tail frames** (zero-padded
overhang) so no trailing samples are left uncovered; once a second interior
frame exists a single **head frame** starting at ``-frame_length // 2`` is
prepended — the windowed first interior frame contributes nothing at sample
0, so without it the leading samples could never be reconstructed.

Frame centers therefore run ``-(N/2 - H)/fs, (H - N/2)/fs, ...``: the first
*interior* frame is always centered on ``t = 0`` and the last interior frame
on ``t = (L - N)/fs``, so every column's time coordinate is its window's
physical center. When ``L == frame_length`` there is exactly one (interior)
frame and the heatmap has a single column, as the product spec requires.

Perfect reconstruction
----------------------
The inverse is weighted overlap-add: each frame's window is used as the
synthesis window and the output is divided sample-by-sample by the accumulated
squared window. With a window/hop pair satisfying the COLA (constant
overlap-add) condition — Hann at 50 % overlap (``hop = N/2``) being the
textbook case — the denominator is constant and the original signal is
recovered to numerical precision (the head/tail context frames make the
boundaries exact as well). Without a COLA pair (e.g. Hann with no overlap)
a small boundary residual is unavoidable and reported honestly.

The forward transform reuses :func:`app.dft.dft` (the shared hand-written
FFT/DFT kernel); no second transform implementation exists here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import ALLOWED_FRAME_N
from .dft import dft, dft_batch, idft
from .errors import BadRequest
from .windows import get_window


# --------------------------------------------------------------------- framing


def check_frame_params(frame_length: int, hop_length: int) -> None:
    """Validate the product-level framing rules.

    * ``frame_length`` must be one of :data:`ALLOWED_FRAME_N`;
    * ``hop_length`` must be a positive integer not exceeding the frame
      length (a larger hop would leave uncovered samples in between — the
      picture would lie).
    """
    if isinstance(frame_length, bool) or not isinstance(frame_length, int):
        raise BadRequest("frame_length must be an integer")
    if isinstance(hop_length, bool) or not isinstance(hop_length, int):
        raise BadRequest("hop_length must be an integer")
    if frame_length not in ALLOWED_FRAME_N:
        allowed = ", ".join(str(v) for v in sorted(ALLOWED_FRAME_N))
        raise BadRequest(
            f"frame_length must be one of the allowed sizes {allowed}; "
            f"got {frame_length}"
        )
    if hop_length <= 0:
        raise BadRequest(
            f"hop_length must be positive (> 0); got {hop_length}"
        )
    if hop_length > frame_length:
        raise BadRequest(
            f"hop_length ({hop_length}) must not exceed the frame length "
            f"({frame_length}); otherwise samples between frames are skipped"
        )


def frame_starts(num_samples: int, frame_length: int, hop_length: int) -> list[int]:
    """Start indices (possibly negative) of every analysis frame.

    Pure geometric helper: does not validate the allowed-size gear, only the
    positive-hop / no-skip relationship, so it can be unit-tested at any
    length.
    """
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if hop_length > frame_length:
        raise ValueError("hop_length must not exceed frame_length")

    if num_samples < frame_length:
        raise BadRequest(
            f"signal is too short to fill one frame: {num_samples} samples "
            f"for a frame length of {frame_length}"
        )

    starts: list[int] = []
    # Interior frames: s = 0, H, 2H, ... while the whole frame fits.
    last_interior = num_samples - frame_length
    k = 0
    while k * hop_length <= last_interior:
        starts.append(k * hop_length)
        k += 1

    if len(starts) == 1 and num_samples == frame_length:
        # The documented degenerate case: the signal exactly fills one frame,
        # so the heatmap has exactly one column (no head/tail context).
        return starts

    # Context frames for every multi-column layout:
    # Head — the first interior frame contributes nothing at sample 0 under a
    # periodic window, so without it the leading samples could never be
    # reconstructed. It is added as soon as the layout extends beyond a
    # single interior frame (i.e. whenever tail coverage exists too),
    # keeping the two boundaries symmetric.
    if len(starts) >= 2 or num_samples > frame_length:
        starts.append(-(frame_length // 2))
    # Tail — starts beyond the last interior frame that still overlap the
    # signal by at least one sample (overhang zero-padded), so no trailing
    # samples are left uncovered.
    s = k * hop_length
    while s < num_samples:
        starts.append(s)
        s += hop_length

    return sorted(starts)


def extract_frames(
    signal: NDArray[np.float64],
    frame_length: int,
    hop_length: int,
) -> NDArray[np.float64]:
    """Return a ``(num_frames, frame_length)`` matrix of frame samples.

    Frames whose start is negative (head context) or runs past the signal
    (tail context) are zero-padded.
    """
    x = np.asarray(signal, dtype=np.float64)
    starts = frame_starts(x.shape[0], frame_length, hop_length)
    frames = np.zeros((len(starts), frame_length), dtype=np.float64)
    for col, s in enumerate(starts):
        lo = max(0, s)
        hi = min(x.shape[0], s + frame_length)
        frames[col, lo - s : hi - s] = x[lo:hi]
    return frames


# ---------------------------------------------------------------- transforms


def stft(
    signal: NDArray[np.number],
    frame_length: int,
    hop_length: int,
    window_name: str = "hann",
    beta: float | None = None,
) -> dict:
    """Short-time Fourier transform of a real sampled signal.

    Returns a dict with:

    * ``frames`` — complex STFT matrix, shape ``(num_frames, frame_length)``;
    * ``times`` — window-center time of each column in seconds;
    * ``starts`` — integer start index of each frame (context frames < 0 or
      straddling the end);
    * ``window`` — the length-``frame_length`` window coefficients used;
    * ``frame_length``, ``hop_length``.
    """
    check_frame_params(frame_length, hop_length)
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise BadRequest("signal must be a 1-D list of numbers")
    if not np.all(np.isfinite(x)):
        raise BadRequest("signal contains non-finite values (NaN/Infinity)")

    window = get_window(window_name, frame_length, beta)
    starts = frame_starts(x.shape[0], frame_length, hop_length)
    segments = extract_frames(x, frame_length, hop_length)
    windowed = segments * window[np.newaxis, :]

    # Few frames: per-frame FFT is cheapest. Many frames (dense overlap):
    # one batched matrix product with the same DFT kernel coefficients.
    if len(starts) >= 8:
        spectrum = dft_batch(windowed)
    else:
        spectrum = np.empty((len(starts), frame_length), dtype=np.complex128)
        for col in range(len(starts)):
            spectrum[col, :] = dft(windowed[col, :])

    times = (np.asarray(starts, dtype=np.float64) + frame_length / 2.0)
    return {
        "frames": spectrum,
        "times": times,
        "starts": np.asarray(starts, dtype=np.int64),
        "window": window,
        "frame_length": frame_length,
        "hop_length": hop_length,
    }


def istft(
    frames: NDArray[np.complexfloating],
    frame_length: int,
    hop_length: int,
    window: NDArray[np.number] | str = "hann",
    beta: float | None = None,
    signal_length: int | None = None,
    starts: NDArray[np.integer] | None = None,
) -> NDArray[np.float64]:
    """Inverse short-time Fourier transform via weighted overlap-add.

    Each column is inverse-transformed, multiplied by the synthesis window
    (the analysis window by default), accumulated into the time domain and
    divided by the accumulated squared window. Frames are placed at the same
    frame grid as :func:`stft` (or at the explicit ``starts``). ``frames``
    must carry all ``frame_length`` bins — a one-sided/half matrix is a
    rejected structural mismatch, not silently zero-padded.
    """
    check_frame_params(frame_length, hop_length)

    z = np.asarray(frames)
    if z.ndim != 2 or z.shape[0] == 0:
        raise BadRequest(
            "frames must be a non-empty 2-D matrix with one spectrum per frame"
        )
    if z.shape[1] != frame_length:
        raise BadRequest(
            f"each frame has {z.shape[1]} bins but frame_length is "
            f"{frame_length}; cannot reassemble a frame structure that does "
            "not match the forward analysis"
        )
    if not np.all(np.isfinite(z)):
        raise BadRequest("frames contain non-finite values (NaN/Infinity)")

    if isinstance(window, str):
        win = get_window(window, frame_length, beta)
    else:
        win = np.asarray(window, dtype=np.float64)
        if win.shape != (frame_length,):
            raise BadRequest(
                f"synthesis window length ({win.shape[0]}) must equal the "
                f"frame length ({frame_length})"
            )

    if starts is None:
        # The caller must tell us how long the original signal was whenever
        # context frames are involved; infer only for the plain grid case.
        num_frames = z.shape[0]
        if signal_length is None:
            signal_length = frame_length + (num_frames - 1) * hop_length
        pos = frame_starts(signal_length, frame_length, hop_length)
        if len(pos) != num_frames:
            raise BadRequest(
                f"frame structure mismatch: got {num_frames} frames but the "
                f"parameters frame_length={frame_length}, hop_length="
                f"{hop_length}, signal_length={signal_length} imply "
                f"{len(pos)} frames"
            )
        starts_arr = np.asarray(pos, dtype=np.int64)
    else:
        starts_arr = np.asarray(starts, dtype=np.int64)
        if starts_arr.shape != (z.shape[0],):
            raise BadRequest(
                "frame structure mismatch: number of frame positions "
                f"({starts_arr.shape[0]}) does not match the number of "
                f"spectra ({z.shape[0]})"
            )
        if signal_length is None:
            signal_length = int(starts_arr.max()) + frame_length

    if signal_length < frame_length:
        raise BadRequest(
            f"signal_length ({signal_length}) is shorter than one frame "
            f"({frame_length})"
        )

    output = np.zeros(signal_length, dtype=np.float64)
    weight = np.zeros(signal_length, dtype=np.float64)
    for col, s in enumerate(starts_arr):
        segment = np.real_if_close(idft(z[col, :]), tol=1000)
        segment = np.real(segment).astype(np.float64) * win
        lo = max(0, int(s))
        hi = min(signal_length, int(s) + frame_length)
        if hi <= lo:
            continue
        output[lo:hi] += segment[lo - int(s) : hi - int(s)]
        weight[lo:hi] += win[lo - int(s) : hi - int(s)] ** 2

    covered = weight > 1e-12
    output[covered] /= weight[covered]
    # Samples no window ever touched stay zero (they have no representation
    # in the supplied frames); this can only happen at the very edges with a
    # rectangular/no-overlap setup.
    return output
