// Perceptually-ordered heat colormap (an inferno approximation) used by the
// spectrogram: dark purple for silence, through magenta/red, to bright
// yellow for the strongest energy. Pure function of t in [0, 1].

type Rgb = [number, number, number];

const STOPS: Array<[number, Rgb]> = [
  [0.0, [0, 0, 4]],
  [0.2, [40, 11, 84]],
  [0.4, [101, 21, 110]],
  [0.6, [186, 54, 85]],
  [0.8, [249, 140, 10]],
  [1.0, [252, 255, 164]],
];

export function inferno(t: number): Rgb {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < STOPS.length; i++) {
    const [x1, c1] = STOPS[i];
    if (x <= x1) {
      const [x0, c0] = STOPS[i - 1];
      const a = (x - x0) / (x1 - x0);
      return [
        Math.round(c0[0] + a * (c1[0] - c0[0])),
        Math.round(c0[1] + a * (c1[1] - c0[1])),
        Math.round(c0[2] + a * (c1[2] - c0[2])),
      ];
    }
  }
  return STOPS[STOPS.length - 1][1];
}
