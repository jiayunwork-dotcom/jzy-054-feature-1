// Shared types mirroring the backend API contracts
// (backend/app/schemas.py).

export type WaveformKind = 'sine' | 'cosine' | 'square' | 'triangle' | 'sawtooth';

export interface Component {
  id: number;
  enabled: boolean;
  kind: WaveformKind;
  amplitude: number;
  frequency: number; // Hz
  phase: number; // radians
}

export type WindowName = 'rect' | 'hann' | 'hamming' | 'blackman' | 'kaiser';

export interface WindowSelection {
  name: WindowName;
  beta: number | null;
}

export interface Spectrum {
  window: WindowName;
  beta: number | null;
  real: number[];
  imag: number[];
  magnitude: number[];
  phase: number[];
  power: number[];
  magnitude_db: number[];
}

export interface DftResponse {
  n: number;
  n_padded: number;
  signal_length: number;
  fs: number;
  frequencies: number[];
  spectra: Spectrum[];
}

export interface WindowInfo {
  name: WindowName;
  beta: number | null;
  coefficients: number[];
  mainlobe_bins: number | null;
  peak_sidelobe_db: number | null;
}

export interface FilterResponse {
  filtered_signal: number[];
  spectrum_real: number[];
  spectrum_imag: number[];
  mask: number[];
  frequencies: number[];
  input_energy: number;
  output_energy: number;
  residual_high_energy: number;
}

export interface SamplingResponse {
  original_t: number[];
  original_y: number[];
  sample_t: number[];
  sample_y: number[];
  reconstructed_t: number[];
  reconstructed_y: number[];
  aliased: boolean;
  apparent_freq_hz: number;
  nyquist_hz: number;
}

export type FilterMode = 'lowpass' | 'highpass' | 'bandpass';

export interface AppState {
  signal: number[];
  components: Component[];
  drawMode: boolean;
  n: number;
  fs: number;
  paddedN: number | null;
  windows: WindowSelection[];
  filteredSignal: number[] | null;
}
