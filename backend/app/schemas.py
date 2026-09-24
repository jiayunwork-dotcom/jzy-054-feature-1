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


class StftRequest(BaseModel):
    signal: list[float]
    fs: float = Field(description="sampling rate in Hz")
    frame_length: int = Field(
        description="samples per segment; one of 32/64/128/256/512"
    )
    hop_length: int = Field(
        description="samples advanced between consecutive frames; 1..frame_length"
    )
    window: WindowSpec = Field(
        default_factory=lambda: WindowSpec(name="hann"),
        description="window applied to each frame",
    )
    include_spectrum: bool = Field(
        default=False,
        description="also return the full complex STFT matrix needed by /istft",
    )


class StftResponse(BaseModel):
    frame_length: int
    hop_length: int
    num_frames: int
    signal_length: int
    fs: float
    window: str
    beta: float | None = None
    times: list[float]
    frame_starts: list[int]
    frequencies: list[float]
    # One-sided (0..fs/2) display matrix, row-major [time][frequency].
    magnitude: list[list[float]]
    power: list[list[float]]
    time_resolution_s: float
    frequency_resolution_hz: float
    overlap_ratio: float
    # Full-length complex spectra, present only when include_spectrum=true.
    spectra_real: list[list[float]] | None = None
    spectra_imag: list[list[float]] | None = None


class IstftRequest(BaseModel):
    fs: float = Field(description="sampling rate in Hz (echoed back)")
    frame_length: int
    hop_length: int
    signal_length: int = Field(
        description="number of samples of the original signal"
    )
    spectra_real: list[list[float]]
    spectra_imag: list[list[float]]
    window: WindowSpec = Field(
        default_factory=lambda: WindowSpec(name="hann"),
    )


class IstftResponse(BaseModel):
    signal: list[float]
    signal_length: int
