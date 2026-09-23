// Signal synthesis: turn the stacked component list into the discrete sample
// sequence that feeds every downstream frequency-domain operation.

import type { Component, WaveformKind } from '../types';

export const MAX_COMPONENTS = 8;

// Each periodic waveform is generated from its normalized phase
// u in [0, 1): keeps sine/cosine exact while the others avoid the
// discontinuity artifacts of a sign on a rounded argument.
function waveform(kind: WaveformKind, u: number): number {
  switch (kind) {
    case 'sine':
      return Math.sin(2 * Math.PI * u);
    case 'cosine':
      return Math.cos(2 * Math.PI * u);
    case 'square':
      return u < 0.5 ? 1 : -1;
    case 'triangle':
      // 0 -> 1 -> 0 over one period
      return u < 0.5 ? 4 * u - 1 : 3 - 4 * u;
    case 'sawtooth':
      return 2 * u - 1;
  }
}

/**
 * Sample the superposition of all *enabled* components.
 * `n` samples at rate `fs`, phase in radians.
 */
export function synthesize(components: Component[], n: number, fs: number): number[] {
  const out = new Array<number>(n).fill(0);
  const active = components.filter((c) => c.enabled && c.amplitude !== 0);
  const dt = 1 / fs;
  for (let i = 0; i < n; i++) {
    const t = i * dt;
    let v = 0;
    for (const c of active) {
      const u = c.frequency * t + c.phase / (2 * Math.PI);
      v += c.amplitude * waveform(c.kind, u - Math.floor(u));
    }
    out[i] = v;
  }
  return out;
}

/**
 * Discretize a freehand stroke. The user paints values against a normalized
 * x in [0, 1] (canvas fraction); we linearly resample that polyline at the
 * n sample instants. Exactly the current sampling settings are used, as the
 * spec requires.
 */
export function resampleStroke(strokes: Array<[number, number]>, n: number): number[] {
  if (strokes.length === 0) return new Array<number>(n).fill(0);
  // A single drag produces one stroke; guard anyway by sorting and
  // monotonizing x (keep last value at a given x).
  const pts = strokes
    .map(([x, y]) => [x, y] as [number, number])
    .sort((a, b) => a[0] - b[0]);
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const x = n === 1 ? 0 : i / (n - 1);
    if (x <= xs[0]) out[i] = ys[0];
    else if (x >= xs[xs.length - 1]) out[i] = ys[ys.length - 1];
    else {
      // binary search the surrounding segment
      let lo = 0;
      let hi = xs.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (xs[mid] <= x) lo = mid;
        else hi = mid;
      }
      const span = xs[hi] - xs[lo] || 1;
      const a = (x - xs[lo]) / span;
      out[i] = ys[lo] * (1 - a) + ys[hi] * a;
    }
  }
  return out;
}
