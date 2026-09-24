// Thin client for the FastAPI arithmetic backend. Every failed call surfaces
// the server's explanatory message (HTTP 400 carries {"detail": "..."}).

import type {
  DftResponse,
  FilterMode,
  FilterResponse,
  IstftResponse,
  SamplingResponse,
  StftResponse,
  WindowInfo,
  WindowSelection,
} from './types';

async function post<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(`无法连接后端（${path}），请确认 API 服务已启动`);
  }
  if (!res.ok) {
    let message = `后端返回 ${res.status}`;
    try {
      const data = await res.json();
      if (typeof data.detail === 'string') message = data.detail;
    } catch {
      /* keep generic message */
    }
    throw new Error(message);
  }
  return (await res.json()) as T;
}

export const api = {
  dft(
    signal: number[],
    fs: number,
    n: number,
    paddedN: number | null,
    windows: WindowSelection[],
  ): Promise<DftResponse> {
    return post('/api/dft', { signal, fs, n, padded_n: paddedN, windows });
  },

  idft(real: number[], imag: number[]): Promise<{ signal: number[] }> {
    return post('/api/idft', { real, imag });
  },

  windows(n: number, selections: WindowSelection[]): Promise<{ windows: WindowInfo[] }> {
    return post('/api/windows', { n, windows: selections });
  },

  filter(
    signal: number[],
    fs: number,
    mode: FilterMode,
    cutoffLow: number | null,
    cutoffHigh: number | null,
  ): Promise<FilterResponse> {
    return post('/api/filter', {
      signal,
      fs,
      mode,
      cutoff_low: cutoffLow,
      cutoff_high: cutoffHigh,
    });
  },

  sampling(
    signalFreq: number,
    fs: number,
    nSamples = 24,
    phase = 0,
  ): Promise<SamplingResponse> {
    return post('/api/sampling', {
      signal_freq: signalFreq,
      fs,
      n_samples: nSamples,
      phase,
    });
  },

  stft(
    signal: number[],
    fs: number,
    frameLength: number,
    hopLength: number,
    window: WindowSelection,
    includeSpectrum = false,
  ): Promise<StftResponse> {
    return post('/api/stft', {
      signal,
      fs,
      frame_length: frameLength,
      hop_length: hopLength,
      window,
      include_spectrum: includeSpectrum,
    });
  },

  istft(
    spectraReal: number[][],
    spectraImag: number[][],
    frameLength: number,
    hopLength: number,
    signalLength: number,
    fs: number,
    window: WindowSelection,
  ): Promise<IstftResponse> {
    return post('/api/istft', {
      spectra_real: spectraReal,
      spectra_imag: spectraImag,
      frame_length: frameLength,
      hop_length: hopLength,
      signal_length: signalLength,
      fs,
      window,
    });
  },
};
