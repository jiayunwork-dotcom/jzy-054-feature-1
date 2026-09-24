"""Pydantic request/response schemas for the API layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WindowSpec(BaseModel):
    name: str
    beta: float | None = None


class DftRequest(BaseModel):
    signal: list[float]
    fs: float = Field(description="sampling rate in Hz")
    n: int = Field(description="selected DFT size, one of 64/128/256/512/1024")
    padded_n: int | None = Field(
        default=None, description="zero-padded transform length (>= signal length)"
    )
    windows: list[WindowSpec] = Field(
        default_factory=lambda: [WindowSpec(name="rect")],
        description="windows to apply before the transform; spectra are "
        "returned in the same order",
    )


class Spectrum(BaseModel):
    window: str
    beta: float | None = None
    real: list[float]
    imag: list[float]
    magnitude: list[float]
    phase: list[float]
    power: list[float]
    magnitude_db: list[float]


class DftResponse(BaseModel):
    n: int
    n_padded: int
    signal_length: int
    fs: float
    frequencies: list[float]
    spectra: list[Spectrum]


class IdftRequest(BaseModel):
    real: list[float]
    imag: list[float] | None = None


class IdftResponse(BaseModel):
    signal: list[float]


class WindowRequest(BaseModel):
    n: int
    windows: list[WindowSpec]


class WindowInfo(BaseModel):
    name: str
    beta: float | None = None
    coefficients: list[float]
    mainlobe_bins: float | None = None
    peak_sidelobe_db: float | None = None


class WindowResponse(BaseModel):
    windows: list[WindowInfo]


class FilterRequest(BaseModel):
    signal: list[float]
    fs: float
    mode: str = Field(description="lowpass | highpass | bandpass")
    cutoff_low: float | None = None
    cutoff_high: float | None = None


class FilterResponse(BaseModel):
    filtered_signal: list[float]
    spectrum_real: list[float]
    spectrum_imag: list[float]
    mask: list[int]
    frequencies: list[float]
    input_energy: float
    output_energy: float
    residual_high_energy: float


class SamplingRequest(BaseModel):
    signal_freq: float
    fs: float
    n_samples: int = 24
    phase: float = 0.0


class SamplingResponse(BaseModel):
    original_t: list[float]
    original_y: list[float]
    sample_t: list[float]
    sample_y: list[float]
    reconstructed_t: list[float]
    reconstructed_y: list[float]
    aliased: bool
    apparent_freq_hz: float
    nyquist_hz: float


# ------------------------------------------------------------------ STFT ---


class StftRequest(BaseModel):
    signal: list[float]
    fs: float = Field(description="sampling rate in Hz")
    frame_length: int = Field(
        description="samples per analysis frame; one of 64/128/256/512/1024"
    )
    hop: int = Field(
        description="frame advance in samples; 1 <= hop <= frame_length "
        "(smaller = more overlap = smoother time axis, more frames to compute)"
    )
    window: WindowSpec = Field(
        default_factory=lambda: WindowSpec(name="hann"),
        description="analysis window applied to every frame",
    )


class StftResponse(BaseModel):
    frame_length: int
    hop: int
    num_frames: int
    fs: float
    window: str
    beta: float | None = None
    # Axes: one entry per column (time center, seconds) and per row
    # (one-sided frequency 0 .. fs/2, Hz).
    times: list[float]
    frequencies: list[float]
    # Intensity matrices, shape (num_frequencies, num_frames): rows =
    # frequency, columns = time.
    power: list[list[float]]
    magnitude: list[list[float]]
    magnitude_db: list[list[float]]
    # Time/frequency resolution figures surfaced for the trade-off readout.
    time_resolution_s: float = Field(
        description="duration spanned by one column: hop / fs (seconds)"
    )
    frame_duration_s: float = Field(
        description="duration covered by one frame: frame_length / fs (seconds)"
    )
    frequency_resolution_hz: float = Field(
        description="spacing of two neighboring frequency rows: fs / frame_length"
    )
    overlap_ratio: float = Field(
        description="fraction of adjacent frames shared: 1 - hop/frame_length"
    )


class IstftRequest(BaseModel):
    # Full length-frame_length spectra, one row per analysis frame (all bins,
    # not just the one-sided view), in the convention returned by the STFT
    # module. signal_length pins the expected output so frame-structure
    # mismatches are rejected explicitly.
    real: list[list[float]]
    imag: list[list[float]] | None = None
    frame_length: int
    hop: int
    signal_length: int
    window: WindowSpec = Field(
        default_factory=lambda: WindowSpec(name="hann"),
    )


class IstftResponse(BaseModel):
    signal: list[float]
    num_frames: int
