// Small dependency-free 2D plotting helper built directly on Canvas 2D.
// Provides: HiDPI-aware sizing, axis box + grid + numeric labels, and
// helpers for line traces, point markers and highlighted spans.

export interface Range {
  min: number;
  max: number;
}

export interface PlotOptions {
  xLabel?: string;
  yLabel?: string;
  yRange?: Range | null;
  gridCount?: number;
}

interface Padding {
  l: number;
  r: number;
  t: number;
  b: number;
}

const FONT = '11px ui-monospace, Menlo, Consolas, monospace';

function niceTicks(min: number, max: number, count: number): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const span = max - min;
  const step0 = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 1e-9; v += step) ticks.push(v);
  return ticks;
}

export class Plot {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private pad: Padding = { l: 52, r: 12, t: 14, b: 30 };
  private xRange: Range = { min: 0, max: 1 };
  private yRange: Range = { min: -1, max: 1 };
  private fixedY: Range | null = null;
  private xLabel = '';
  private yLabel = '';

  constructor(container: HTMLElement, height = 200, opts: PlotOptions = {}) {
    this.canvas = document.createElement('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = `${height}px`;
    this.canvas.style.display = 'block';
    container.appendChild(this.canvas);
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context unavailable');
    this.ctx = ctx;
    this.fixedY = opts.yRange ?? null;
    this.xLabel = opts.xLabel ?? '';
    this.yLabel = opts.yLabel ?? '';
    this.resize();
  }

  resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth || 300;
    const h = this.canvas.clientHeight || 200;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  get width(): number {
    return this.canvas.clientWidth;
  }
  get height(): number {
    return this.canvas.clientHeight;
  }

  // Data coordinates -> CSS pixel coordinates within the axis box.
  sx(x: number): number {
    const { l, r } = this.pad;
    const w = this.width - l - r;
    return l + ((x - this.xRange.min) / (this.xRange.max - this.xRange.min)) * w;
  }
  sy(y: number): number {
    const { t, b } = this.pad;
    const h = this.height - t - b;
    return t + h - ((y - this.yRange.min) / (this.yRange.max - this.yRange.min)) * h;
  }
  pxToDataX(px: number): number {
    const { l, r } = this.pad;
    const w = this.width - l - r;
    return this.xRange.min + ((px - l) / w) * (this.xRange.max - this.xRange.min);
  }

  pxToDataY(py: number): number {
    const { t, b } = this.pad;
    const h = this.height - t - b;
    return this.yRange.min + (1 - (py - t) / h) * (this.yRange.max - this.yRange.min);
  }

  /**
   * Convert mouse client coordinates to data coordinates. Returns null when
   * the pointer is outside the axis box.
   */
  toData(clientX: number, clientY: number): { x: number; y: number } | null {
    const rect = this.canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const { l, r, t, b } = this.pad;
    if (px < l || px > this.width - r || py < t || py > this.height - b) return null;
    return { x: this.pxToDataX(px), y: this.pxToDataY(py) };
  }

  /** Fix (or release) the y-range used for every subsequent frame. */
  setFixedY(range: Range | null): void {
    this.fixedY = range;
  }

  beginFrame(xRange: Range, yRange?: Range | null): void {
    this.xRange = xRange;
    if (this.fixedY) this.yRange = this.fixedY;
    else if (yRange) this.yRange = yRange;
    else this.yRange = { min: -1, max: 1 };
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);
    this.drawGrid();
  }

  private drawGrid(): void {
    const ctx = this.ctx;
    const { l, r, t, b } = this.pad;
    const x0 = l;
    const x1 = this.width - r;
    const y0 = t;
    const y1 = this.height - b;

    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.fillStyle = 'rgba(230,235,245,0.65)';
    ctx.font = FONT;
    ctx.lineWidth = 1;

    const xt = niceTicks(this.xRange.min, this.xRange.max, 6);
    for (const v of xt) {
      const px = this.sx(v);
      if (px < x0 - 0.5 || px > x1 + 0.5) continue;
      ctx.beginPath();
      ctx.moveTo(px, y0);
      ctx.lineTo(px, y1);
      ctx.stroke();
      const label = Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(v < 10 ? 2 : 0);
      ctx.textAlign = 'center';
      ctx.fillText(label, px, y1 + 14);
    }
    const yt = niceTicks(this.yRange.min, this.yRange.max, 4);
    for (const v of yt) {
      const py = this.sy(v);
      if (py < y0 - 0.5 || py > y1 + 0.5) continue;
      ctx.beginPath();
      ctx.moveTo(x0, py);
      ctx.lineTo(x1, py);
      ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(v.toFixed(2), x0 - 6, py + 3);
    }

    // Zero lines emphasized.
    ctx.strokeStyle = 'rgba(255,255,255,0.28)';
    if (0 >= this.yRange.min && 0 <= this.yRange.max) {
      ctx.beginPath();
      ctx.moveTo(x0, this.sy(0));
      ctx.lineTo(x1, this.sy(0));
      ctx.stroke();
    }

    // Axis frame.
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

    ctx.fillStyle = 'rgba(200,210,230,0.85)';
    if (this.xLabel) {
      ctx.textAlign = 'center';
      ctx.fillText(this.xLabel, (x0 + x1) / 2, this.height - 2);
    }
    if (this.yLabel) {
      ctx.save();
      ctx.translate(12, (y0 + y1) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText(this.yLabel, 0, 0);
      ctx.restore();
    }
  }

  vline(x: number, color: string, dash: number[] = [], width = 1): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.setLineDash(dash);
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(this.sx(x), this.pad.t);
    ctx.lineTo(this.sx(x), this.height - this.pad.b);
    ctx.stroke();
    ctx.restore();
  }

  hspan(x0: number, x1: number, color: string): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = color;
    const px0 = this.sx(Math.max(x0, this.xRange.min));
    const px1 = this.sx(Math.min(x1, this.xRange.max));
    ctx.fillRect(px0, this.pad.t, Math.max(0, px1 - px0), this.height - this.pad.t - this.pad.b);
    ctx.restore();
  }

  line(xs: ArrayLike<number>, ys: ArrayLike<number>, color: string, width = 1.6, dash: number[] = []): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i];
      const y = ys[i];
      if (!Number.isFinite(y)) {
        started = false;
        continue;
      }
      const px = this.sx(x);
      const py = this.sy(y);
      if (!started) {
        ctx.moveTo(px, py);
        started = true;
      } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
  }

  points(xs: ArrayLike<number>, ys: ArrayLike<number>, color: string, radius = 2.5): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = color;
    for (let i = 0; i < xs.length; i++) {
      ctx.beginPath();
      ctx.arc(this.sx(xs[i]), this.sy(ys[i]), radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  text(str: string, x: number, y: number, color = 'rgba(230,235,245,0.9)', align: CanvasTextAlign = 'left'): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = `12px ui-sans-serif, system-ui, sans-serif`;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.fillText(str, this.sx(x), this.sy(y));
    ctx.restore();
  }
}

export function autoRange(values: ArrayLike<number>, padRatio = 0.08): Range {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { min: -1, max: 1 };
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const pad = (hi - lo) * padRatio;
  return { min: lo - pad, max: hi + pad };
}
