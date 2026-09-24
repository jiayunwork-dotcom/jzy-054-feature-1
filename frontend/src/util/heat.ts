// Dependency-free perceptually-smooth heat color map for the spectrogram.
// A compact set of anchor colors (dark blue -> purple -> red -> orange ->
// bright yellow, inspired by matplotlib's "inferno") is linearly
// interpolated into a 256-entry lookup table; no third-party charting.

interface RGB {
  r: number;
  g: number;
  b: number;
}

const ANCHORS: Array<[number, RGB]> = [
  [0.0, { r: 4, g: 4, b: 20 }],
  [0.15, { r: 38, g: 16, b: 70 }],
  [0.35, { r: 95, g: 30, b: 110 }],
  [0.55, { r: 165, g: 45, b: 90 }],
  [0.72, { r: 225, g: 90, b: 45 }],
  [0.86, { r: 245, g: 160, b: 40 }],
  [1.0, { r: 252, g: 244, b: 170 }],
];

function buildLut(): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const v = i / 255;
    let a = ANCHORS[0];
    let b2 = ANCHORS[ANCHORS.length - 1];
    for (let j = 0; j < ANCHORS.length - 1; j++) {
      if (v >= ANCHORS[j][0] && v <= ANCHORS[j + 1][0]) {
        a = ANCHORS[j];
        b2 = ANCHORS[j + 1];
        break;
      }
    }
    const span = b2[0] - a[0] || 1;
    const t = (v - a[0]) / span;
    lut[i * 3] = a[1].r + (b2[1].r - a[1].r) * t;
    lut[i * 3 + 1] = a[1].g + (b2[1].g - a[1].g) * t;
    lut[i * 3 + 2] = a[1].b + (b2[1].b - a[1].b) * t;
  }
  return lut;
}

export const HEAT_LUT = buildLut();

/** Map a normalized [0,1] value to a CSS rgb() string. */
export function heatColor(v: number): string {
  const i = Math.max(0, Math.min(255, Math.round(v * 255)));
  return `rgb(${HEAT_LUT[i * 3]},${HEAT_LUT[i * 3 + 1]},${HEAT_LUT[i * 3 + 2]})`;
}
