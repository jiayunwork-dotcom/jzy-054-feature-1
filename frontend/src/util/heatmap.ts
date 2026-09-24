// Dedicated 2-D time-frequency heatmap drawn directly on Canvas 2D (no
// third-party charting library). The spectrogram matrix is painted into an
// offscreen ImageData one cell per pixel and scaled onto the axis box with
// smoothing on, so dense matrices render as a smooth image while small ones
// (a single time column stays a single column) still show blocky cells.
//
// Data convention: matrix[col][row] where columns are time (left -> right)
// and rows are frequency (row 0 = DC at the BOTTOM, last row = fs/2 on top).

import { HEAT_LUT } from './heat';
import { PLOT_FONT, niceTicks } from './plot';

export interface HeatmapData {
  // matrix[timeIndex][frequencyIndex], dB values
  db: Float32Array;
  cols: number;
  rows: number;
  times: number[]; // seconds per column
  frequencies: number[]; // Hz per row
  tMin: number; // first column center
  tMax: number; // last column center
  axisMin: number; // axis extent, padded for a single column
  axisMax: number;
  fMax: number;
  dbMin: number;
  dbMax: number;
}

interface Padding {
  l: number;
  r: number;
  t: number;
  b: number;
}

export interface HoverInfo {
  time: number;
  frequency: number;
  db: number;
  magnitude: number;
  col: number;
  row: number;
  px: number;
  py: number;
}

const PAD: Padding = { l: 56, r: 64, t: 12, b: 32 };

export class Heatmap {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private data: HeatmapData | null = null;
  private offscreen: HTMLCanvasElement;
  private offCtx: CanvasRenderingContext2D;
  // Base frame (image + axes + color bar) without the hover overlay, cached
  // so mousemove only restores it and composites the crosshair/chip.
  private base: HTMLCanvasElement;
  private baseCtx: CanvasRenderingContext2D;

  constructor(container: HTMLElement, height = 300) {
    this.canvas = document.createElement('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = `${height}px`;
    this.canvas.style.display = 'block';
    container.appendChild(this.canvas);
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context unavailable');
    this.ctx = ctx;
    this.offscreen = document.createElement('canvas');
    const offCtx = this.offscreen.getContext('2d');
    if (!offCtx) throw new Error('offscreen Canvas 2D context unavailable');
    this.offCtx = offCtx;
    this.base = document.createElement('canvas');
    const baseCtx = this.base.getContext('2d');
    if (!baseCtx) throw new Error('base Canvas 2D context unavailable');
    this.baseCtx = baseCtx;
    this.resize();
  }

  resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth || 300;
    const h = this.canvas.clientHeight || 300;
    for (const c of [this.canvas, this.base]) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.baseCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  get width(): number {
    return this.canvas.clientWidth;
  }
  get height(): number {
    return this.canvas.clientHeight;
  }

  private box() {
    return {
      x0: PAD.l,
      y0: PAD.t,
      w: this.width - PAD.l - PAD.r,
      h: this.height - PAD.t - PAD.b,
    };
  }

  /** Time (s) -> CSS px within the axis box. */
  private timeToPx(t: number): number {
    const box = this.box();
    const d = this.data!;
    return box.x0 + ((t - d.axisMin) / (d.axisMax - d.axisMin)) * box.w;
  }

  setData(data: HeatmapData): void {
    this.data = data;
    this.render();
  }
  pick(clientX: number, clientY: number): HoverInfo | null {
    const d = this.data;
    if (!d || d.cols === 0 || d.rows === 0) return null;
    const rect = this.canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const box = this.box();
    if (px < box.x0 || px > box.x0 + box.w || py < box.y0 || py > box.y0 + box.h) {
      return null;
    }
    const fx = (px - box.x0) / box.w;
    const fy = (py - box.y0) / box.h; // 0 at top
    const t = d.axisMin + fx * (d.axisMax - d.axisMin);
    const f = (1 - fy) * d.fMax;
    // Nearest actual cell (columns are evenly spaced by construction).
    const spacing =
      d.cols > 1 ? (d.times[d.cols - 1] - d.times[0]) / (d.cols - 1) : 1;
    let col = d.cols > 1 ? Math.round((t - d.times[0]) / spacing) : 0;
    col = Math.max(0, Math.min(d.cols - 1, col));
    const row = Math.max(
      0,
      Math.min(d.rows - 1, Math.round((f / d.fMax) * (d.rows - 1))),
    );
    const db = d.db[col * d.rows + row];
    return {
      time: d.times[col],
      frequency: d.frequencies[row],
      db,
      magnitude: Math.pow(10, db / 20),
      col,
      row,
      px,
      py,
    };
  }

  render(): void {
    this.drawBase(this.baseCtx);
    this.ctx.clearRect(0, 0, this.width, this.height);
    this.ctx.drawImage(this.base, 0, 0, this.width, this.height);
  }

  private drawBase(ctx: CanvasRenderingContext2D): void {
    ctx.clearRect(0, 0, this.width, this.height);
    const d = this.data;
    const box = this.box();
    if (!d) {
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.strokeRect(box.x0, box.y0, box.w, box.h);
      return;
    }

    // --- image ---------------------------------------------------------
    const { cols, rows } = d;
    if (this.offscreen.width !== cols || this.offscreen.height !== rows) {
      this.offscreen.width = cols;
      this.offscreen.height = rows;
    }
    const img = this.offCtx.createImageData(cols, rows);
    const span = Math.max(1e-9, d.dbMax - d.dbMin);
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        // Stored row 0 = DC (bottom of figure); image row 0 = top.
        const v = (d.db[c * rows + (rows - 1 - r)] - d.dbMin) / span;
        const clamped = v < 0 ? 0 : v > 1 ? 1 : v;
        const li = Math.round(clamped * 255) * 3;
        const oi = (r * cols + c) * 4;
        img.data[oi] = HEAT_LUT[li];
        img.data[oi + 1] = HEAT_LUT[li + 1];
        img.data[oi + 2] = HEAT_LUT[li + 2];
        img.data[oi + 3] = 255;
      }
    }
    this.offCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = cols < box.w / 2 || rows < box.h / 2;
    ctx.imageSmoothingQuality = 'high';
    if (cols === 1) {
      // Single-column degenerate case: draw one centered, blocky column
      // instead of stretching it across the whole axis.
      const colW = Math.max(4, Math.min(40, box.w / 4));
      ctx.drawImage(
        this.offscreen,
        this.timeToPx(d.times[0]) - colW / 2,
        box.y0,
        colW,
        box.h,
      );
    } else {
      // Each column centered on its time coordinate: the first and last
      // column centers sit half a column width inside the box.
      const pFirst = this.timeToPx(d.times[0]);
      const pLast = this.timeToPx(d.times[cols - 1]);
      const colW = (pLast - pFirst) / (cols - 1);
      ctx.drawImage(this.offscreen, pFirst - colW / 2, box.y0, colW * cols, box.h);
    }

    // --- grid + labels -------------------------------------------------
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.fillStyle = 'rgba(230,235,245,0.7)';
    ctx.font = PLOT_FONT;
    ctx.lineWidth = 1;

    // Ticks are labeled only over the physical signal span [0, tMax], even
    // though context columns may be painted at slightly negative times.
    const tickMin = Math.max(0, d.axisMin);
    const tickMax = d.tMax;
    const xTicks = niceTicks(tickMin, tickMax, 6);
    for (const v of xTicks) {
      if (v < tickMin - 1e-9 || v > tickMax + 1e-9) continue;
      const px = this.timeToPx(v);
      ctx.beginPath();
      ctx.moveTo(px, box.y0);
      ctx.lineTo(px, box.y0 + box.h);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(`${v.toFixed(2)}s`, px, box.y0 + box.h + 15);
    }
    const yTicks = niceTicks(0, d.fMax, 5);
    for (const v of yTicks) {
      const py = box.y0 + box.h - (v / d.fMax) * box.h;
      ctx.beginPath();
      ctx.moveTo(box.x0, py);
      ctx.lineTo(box.x0 + box.w, py);
      ctx.stroke();
      ctx.textAlign = 'right';
      const label = Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(v < 10 ? 1 : 0);
      ctx.fillText(`${label} Hz`, box.x0 - 6, py + 3);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.45)';
    ctx.strokeRect(box.x0, box.y0, box.w, box.h);

    // Signal extent markers (t=0 and t=L/fs); context frames can extend past.
    ctx.strokeStyle = 'rgba(255,209,102,0.8)';
    ctx.setLineDash([5, 4]);
    for (const edge of [0, d.tMax]) {
      if (edge < d.axisMin || edge > d.axisMax) continue;
      const px = this.timeToPx(edge);
      ctx.beginPath();
      ctx.moveTo(px, box.y0);
      ctx.lineTo(px, box.y0 + box.h);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Axis titles.
    ctx.fillStyle = 'rgba(200,210,230,0.85)';
    ctx.textAlign = 'center';
    ctx.fillText('时间 t (s)', box.x0 + box.w / 2, this.height - 2);
    ctx.save();
    ctx.translate(13, box.y0 + box.h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('频率 f (Hz)', 0, 0);
    ctx.restore();

    // --- color bar -----------------------------------------------------
    const barW = 12;
    const barX = box.x0 + box.w + 18;
    const grad = ctx.createLinearGradient(0, box.y0 + box.h, 0, box.y0);
    for (let i = 0; i <= 8; i++) {
      const v = i / 8;
      const li = Math.round(v * 255) * 3;
      grad.addColorStop(v, `rgb(${HEAT_LUT[li]},${HEAT_LUT[li + 1]},${HEAT_LUT[li + 2]})`);
    }
    ctx.fillStyle = grad;
    ctx.fillRect(barX, box.y0, barW, box.h);
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.strokeRect(barX, box.y0, barW, box.h);
    ctx.fillStyle = 'rgba(230,235,245,0.7)';
    ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${Math.round(d.dbMax)}`, barX + barW + 4, box.y0 + 4);
    ctx.fillText(`${Math.round(d.dbMin)}`, barX + barW + 4, box.y0 + box.h);
    ctx.save();
    ctx.translate(barX + barW + 26, box.y0 + box.h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('幅度 (dB)', 0, 0);
    ctx.restore();
  }

  /** Draw the hover crosshair + readout chip; pass null to clear. */
  drawHover(info: HoverInfo | null): void {
    // Restore the cached base frame (no matrix re-rasterization here).
    this.ctx.clearRect(0, 0, this.width, this.height);
    this.ctx.drawImage(this.base, 0, 0, this.width, this.height);
    if (!info) return;
    const ctx = this.ctx;
    const box = this.box();
    // Crosshair clipped to the image box.
    ctx.save();
    ctx.beginPath();
    ctx.rect(box.x0, box.y0, box.w, box.h);
    ctx.clip();
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(info.px, box.y0);
    ctx.lineTo(info.px, box.y0 + box.h);
    ctx.moveTo(box.x0, info.py);
    ctx.lineTo(box.x0 + box.w, info.py);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.beginPath();
    ctx.arc(info.px, info.py, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Readout chip.
    const lines = [
      `t = ${info.time.toFixed(3)} s`,
      `f = ${info.frequency.toFixed(2)} Hz`,
      `|X| = ${info.magnitude.toExponential(2)}  (${info.db.toFixed(1)} dB)`,
    ];
    ctx.font = '11px ui-monospace, Menlo, Consolas, monospace';
    const chipW = 168;
    const chipH = 44;
    let cx = info.px + 12;
    let cy = info.py - chipH - 8;
    if (cx + chipW > box.x0 + box.w) cx = info.px - chipW - 12;
    if (cy < box.y0) cy = info.py + 12;
    ctx.fillStyle = 'rgba(10,14,24,0.9)';
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.beginPath();
    ctx.roundRect(cx, cy, chipW, chipH, 6);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = 'rgba(230,235,245,0.95)';
    ctx.textAlign = 'left';
    lines.forEach((line, i) => ctx.fillText(line, cx + 8, cy + 16 + i * 13));
  }
}
