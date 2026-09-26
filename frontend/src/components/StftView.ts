// Time-frequency analysis (short-time Fourier transform): the signal is cut
// into overlapping windowed frames by the backend, and each frame's spectrum
// becomes one column of a 2-D heatmap — x = time, y = frequency, color =
// energy. Three local controls (frame length, hop/overlap, window) reshape
// the time/frequency resolution tradeoff, which is printed as hard numbers
// above the plot. A one-click linear chirp is the built-in test signal: it
// must draw a clean rising diagonal.

import { api } from '../api';
import type { StftResponse, WindowName } from '../types';
import { store } from '../store';
import { niceTicks } from '../util/plot';
import { inferno } from '../util/colormap';

const FRAME_LEN_CHOICES = [64, 128, 256, 512, 1024];
const DYN_RANGE_DB = 80; // color scale spans this many dB below the peak

const WINDOW_LABELS: Record<WindowName, string> = {
  rect: '矩形窗',
  hann: '汉宁窗',
  hamming: '汉明窗',
  blackman: '布莱克曼窗',
  kaiser: '凯泽窗',
};

const FONT = '11px ui-monospace, Menlo, Consolas, monospace';
const PAD = { l: 56, r: 74, t: 10, b: 30 };

export class StftView {
  private root: HTMLElement;
  private frameLenSelect!: HTMLSelectElement;
  private hopSlider!: HTMLInputElement;
  private hopVal!: HTMLElement;
  private windowSelect!: HTMLSelectElement;
  private betaWrap!: HTMLElement;
  private betaInput!: HTMLInputElement;
  private resoEl!: HTMLElement;
  private readoutEl!: HTMLElement;
  private statusEl!: HTMLElement;
  private canvas!: HTMLCanvasElement;
  private ctx!: CanvasRenderingContext2D;
  private off: HTMLCanvasElement;

  private frameLen = 64;
  private hop = 16;
  private data: StftResponse | null = null;
  private maxMag = 1e-12;
  private hover: { j: number; i: number } | null = null;
  private inFlight = 0;
  private debounceTimer: number | null = null;

  constructor(root: HTMLElement) {
    this.root = root;
    this.off = document.createElement('canvas');
    this.renderShell();
    this.wire();
    store.subscribe(() => this.onStoreChange());
    window.addEventListener('resize', () => this.draw());
    this.onStoreChange(); // initial options + fetch
  }

  // ------------------------------------------------------------------- shell

  private renderShell(): void {
    this.root.innerHTML = `
      <div class="row wrap gap stft-controls">
        <label>段长 M
          <select id="stft-frame-len"></select>
        </label>
        <label class="slider-label">步长（每段往前挪动的采样点）
          <input type="range" id="stft-hop" min="1" max="64" step="1" value="16"/>
          <span id="stft-hop-val" class="slider-val"></span>
        </label>
        <label>窗
          <select id="stft-window">
            ${Object.entries(WINDOW_LABELS)
              .map(
                ([v, label]) =>
                  `<option value="${v}" ${v === 'hann' ? 'selected' : ''}>${label}</option>`,
              )
              .join('')}
          </select>
        </label>
        <label id="stft-beta-wrap" hidden>凯泽 β
          <input id="stft-beta" type="number" min="0" step="0.1" value="5"/>
        </label>
        <button class="btn" id="stft-chirp" type="button">⚡ 载入线性扫频测试信号</button>
      </div>
      <div class="stft-reso" id="stft-reso"></div>
      <div class="canvas-wrap" id="stft-canvas-host"></div>
      <div class="stft-readout" id="stft-readout">把鼠标移到热力图上，读取任意点的时刻 / 频率 / 幅度。</div>
      <div class="hint">
        段长拉长 → 频率分得细、时间上抹得平；段长缩短 → 时间利落、频率变糊（不确定性原理）。
        步长小于段长即相邻段重叠：重叠越多画面越顺，但帧数（计算量）也越大。
      </div>
      <div class="status" id="stft-status"></div>`;

    this.frameLenSelect = this.root.querySelector('#stft-frame-len')!;
    this.hopSlider = this.root.querySelector('#stft-hop')!;
    this.hopVal = this.root.querySelector('#stft-hop-val')!;
    this.windowSelect = this.root.querySelector('#stft-window')!;
    this.betaWrap = this.root.querySelector('#stft-beta-wrap')!;
    this.betaInput = this.root.querySelector('#stft-beta')!;
    this.resoEl = this.root.querySelector('#stft-reso')!;
    this.readoutEl = this.root.querySelector('#stft-readout')!;
    this.statusEl = this.root.querySelector('#stft-status')!;

    const host = this.root.querySelector('#stft-canvas-host')!;
    this.canvas = document.createElement('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = '340px';
    this.canvas.style.display = 'block';
    host.appendChild(this.canvas);
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context unavailable');
    this.ctx = ctx;
  }

  private wire(): void {
    this.frameLenSelect.addEventListener('change', () => {
      this.frameLen = parseInt(this.frameLenSelect.value, 10);
      if (this.hop > this.frameLen) this.hop = Math.max(1, this.frameLen >> 2);
      this.hopSlider.max = String(this.frameLen);
      this.hopSlider.value = String(this.hop);
      this.refreshControlLabels();
      this.scheduleFetch();
    });
    this.hopSlider.addEventListener('input', () => {
      this.hop = parseInt(this.hopSlider.value, 10);
      this.refreshControlLabels();
      this.scheduleFetch();
    });
    this.windowSelect.addEventListener('change', () => {
      this.betaWrap.hidden = this.windowSelect.value !== 'kaiser';
      this.scheduleFetch();
    });
    this.betaInput.addEventListener('input', () => this.scheduleFetch());
    this.root.querySelector('#stft-chirp')!.addEventListener('click', () => this.loadChirp());

    this.canvas.addEventListener('mousemove', (e) => this.onHover(e));
    this.canvas.addEventListener('mouseleave', () => {
      this.hover = null;
      this.readoutEl.textContent = '把鼠标移到热力图上，读取任意点的时刻 / 频率 / 幅度。';
      this.draw();
    });
  }

  // -------------------------------------------------------------- controls

  private currentWindow(): { name: WindowName; beta: number | null } {
    const name = this.windowSelect.value as WindowName;
    if (name !== 'kaiser') return { name, beta: null };
    const beta = parseFloat(this.betaInput.value);
    return { name, beta: Number.isFinite(beta) ? beta : 5 };
  }

  private rebuildFrameLenChoices(): void {
    const n = store.state.signal.length;
    const choices = FRAME_LEN_CHOICES.filter((v) => v <= n);
    if (choices.length === 0) return;
    if (!choices.includes(this.frameLen)) {
      this.frameLen = choices[choices.length - 1];
      this.hop = Math.min(this.hop, Math.max(1, this.frameLen >> 2));
    }
    this.hop = Math.min(this.hop, this.frameLen);
    this.frameLenSelect.innerHTML = choices
      .map((v) => `<option value="${v}" ${v === this.frameLen ? 'selected' : ''}>${v}</option>`)
      .join('');
    this.hopSlider.max = String(this.frameLen);
    this.hopSlider.value = String(this.hop);
  }

  private refreshControlLabels(): void {
    const overlap = Math.round((1 - this.hop / this.frameLen) * 100);
    this.hopVal.textContent = `${this.hop} 点（重叠 ${overlap}%）`;

    // The two resolution numbers, straight from the current settings.
    const fs = store.state.fs;
    const frameMs = (this.frameLen / fs) * 1000;
    const hopMs = (this.hop / fs) * 1000;
    const df = fs / this.frameLen;
    this.resoEl.innerHTML =
      `时间分辨率：每列 = 一段 <b>${this.frameLen}</b> 点 ≈ <b>${frameMs.toFixed(1)} ms</b> 的信号` +
      `（帧间隔 ${hopMs.toFixed(1)} ms，重叠 ${overlap}%）　·　` +
      `频率分辨率：每行 = <b>${df.toFixed(2)} Hz</b>（= fs / M）`;
  }

  private loadChirp(): void {
    const { n, fs } = store.state;
    const f0 = fs / 32;
    const f1 = (3 * fs) / 8;
    const duration = n / fs;
    const rate = (f1 - f0) / duration; // Hz per second
    const signal = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      const t = i / fs;
      signal[i] = Math.sin(2 * Math.PI * (f0 * t + 0.5 * rate * t * t));
    }
    store.update({ signal, filteredSignal: null });
    this.statusEl.textContent =
      `已载入线性扫频信号：频率 ${f0.toFixed(0)} → ${f1.toFixed(0)} Hz 匀速上升，` +
      `时频图上应看到一条向右上爬的斜线。`;
  }

  // ---------------------------------------------------------------- fetching

  private onStoreChange(): void {
    this.rebuildFrameLenChoices();
    this.refreshControlLabels();
    this.scheduleFetch();
  }

  private scheduleFetch(): void {
    if (this.debounceTimer != null) window.clearTimeout(this.debounceTimer);
    this.debounceTimer = window.setTimeout(() => void this.fetchData(), 80);
  }

  private async fetchData(): Promise<void> {
    const { signal, fs } = store.state;
    if (signal.length === 0) return;
    if (signal.length < this.frameLen) {
      this.data = null;
      this.statusEl.textContent = `信号长度 ${signal.length} 不足一个段长 ${this.frameLen}。`;
      this.draw();
      return;
    }
    const token = ++this.inFlight;
    this.statusEl.textContent = '时频计算中…';
    try {
      const data = await api.stft(signal, fs, this.frameLen, this.hop, this.currentWindow());
      if (token !== this.inFlight) return;
      this.data = data;
      this.maxMag = 1e-12;
      for (const row of data.magnitude) {
        for (const v of row) if (v > this.maxMag) this.maxMag = v;
      }
      this.draw();
      const overlap = Math.round((1 - data.hop / data.frame_len) * 100);
      this.statusEl.textContent =
        `共 ${data.num_frames} 帧 × ${data.num_bins} 频率行` +
        `（重叠 ${overlap}%；步长越小帧数越多，计算量越大）`;
    } catch (err) {
      if (token === this.inFlight) {
        this.data = null;
        this.statusEl.textContent = `⚠ ${(err as Error).message}`;
        this.draw();
      }
    }
  }

  // ----------------------------------------------------------------- drawing

  private plotRect(): { x0: number; y0: number; w: number; h: number } {
    const w = this.canvas.clientWidth || 600;
    const h = this.canvas.clientHeight || 340;
    return { x0: PAD.l, y0: PAD.t, w: w - PAD.l - PAD.r, h: h - PAD.t - PAD.b };
  }

  private draw(): void {
    const dpr = window.devicePixelRatio || 1;
    const cw = this.canvas.clientWidth || 600;
    const ch = this.canvas.clientHeight || 340;
    this.canvas.width = Math.round(cw * dpr);
    this.canvas.height = Math.round(ch * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.ctx.clearRect(0, 0, cw, ch);
    if (!this.data) return;

    const d = this.data;
    const F = d.num_frames;
    const B = d.num_bins;
    const { x0, y0, w, h } = this.plotRect();

    // Peak-referenced dB scale (peak cached when the data arrived).
    const maxMag = this.maxMag;

    // Paint the F x B matrix into a 1-cell-per-pixel offscreen canvas, then
    // stretch it (no smoothing: each visible block is one real STFT cell).
    this.off.width = F;
    this.off.height = B;
    const offCtx = this.off.getContext('2d')!;
    const img = offCtx.createImageData(F, B);
    for (let j = 0; j < F; j++) {
      for (let i = 0; i < B; i++) {
        const db = 20 * Math.log10(Math.max(d.magnitude[j][i], 1e-12) / maxMag);
        const t = Math.min(1, Math.max(0, (db + DYN_RANGE_DB) / DYN_RANGE_DB));
        const [r, g, b] = inferno(t);
        const p = ((B - 1 - i) * F + j) * 4; // bin 0 (DC) at the bottom
        img.data[p] = r;
        img.data[p + 1] = g;
        img.data[p + 2] = b;
        img.data[p + 3] = 255;
      }
    }
    offCtx.putImageData(img, 0, 0);
    this.ctx.imageSmoothingEnabled = false;
    this.ctx.drawImage(this.off, x0, y0, w, h);

    // Axes in physical units: column j centers on times[j], row i on
    // frequencies[i]; the image spans half a cell beyond the outer centers.
    const dt = d.time_step_s;
    const df = d.freq_step_hz;
    const xMin = d.times[0] - dt / 2;
    const xMax = d.times[F - 1] + dt / 2;
    const yMin = -df / 2;
    const yMax = d.frequencies[B - 1] + df / 2;
    const sx = (t: number) => x0 + ((t - xMin) / (xMax - xMin)) * w;
    const sy = (f: number) => y0 + h - ((f - yMin) / (yMax - yMin)) * h;

    this.drawAxes(sx, sy, xMin, xMax, yMin, yMax);
    this.drawColorbar();

    if (this.hover) {
      // Clamp against the current matrix: a new fetch may have fewer cells.
      const j = Math.min(this.hover.j, F - 1);
      const i = Math.min(this.hover.i, B - 1);
      const tC = d.times[j];
      const fC = d.frequencies[i];
      const ctx = this.ctx;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(tC), y0);
      ctx.lineTo(sx(tC), y0 + h);
      ctx.moveTo(x0, sy(fC));
      ctx.lineTo(x0 + w, sy(fC));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = '#ffd166';
      ctx.strokeRect(sx(tC - dt / 2), sy(fC + df / 2), sx(tC + dt / 2) - sx(tC - dt / 2), sy(fC - df / 2) - sy(fC + df / 2));
      ctx.restore();
    }
  }

  private drawAxes(
    sx: (t: number) => number,
    sy: (f: number) => number,
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
  ): void {
    const ctx = this.ctx;
    const { x0, y0, w, h } = this.plotRect();
    ctx.save();
    ctx.font = FONT;
    ctx.lineWidth = 1;

    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.fillStyle = 'rgba(230,235,245,0.75)';
    for (const v of niceTicks(Math.max(0, xMin), xMax, 7)) {
      const px = sx(v);
      if (px < x0 - 0.5 || px > x0 + w + 0.5) continue;
      ctx.beginPath();
      ctx.moveTo(px, y0);
      ctx.lineTo(px, y0 + h);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(v < 1 ? 2 : 1), px, y0 + h + 14);
    }
    for (const v of niceTicks(Math.max(0, yMin), yMax, 5)) {
      const py = sy(v);
      if (py < y0 - 0.5 || py > y0 + h + 0.5) continue;
      ctx.beginPath();
      ctx.moveTo(x0, py);
      ctx.lineTo(x0 + w, py);
      ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(v.toFixed(0), x0 - 6, py + 3);
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.strokeRect(x0, y0, w, h);

    ctx.fillStyle = 'rgba(200,210,230,0.85)';
    ctx.textAlign = 'center';
    ctx.fillText('时间 t (s)', x0 + w / 2, y0 + h + 26);
    ctx.save();
    ctx.translate(14, y0 + h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('频率 f (Hz)', 0, 0);
    ctx.restore();
    ctx.restore();
  }

  private drawColorbar(): void {
    const ctx = this.ctx;
    const { y0, h } = this.plotRect();
    const cw = this.canvas.clientWidth || 600;
    const barX = cw - PAD.r + 18;
    const barW = 12;
    const steps = 64;
    for (let s = 0; s < steps; s++) {
      const [r, g, b] = inferno(1 - s / (steps - 1));
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(barX, y0 + (s * h) / steps, barW, h / steps + 1);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.strokeRect(barX, y0, barW, h);
    ctx.fillStyle = 'rgba(230,235,245,0.75)';
    ctx.font = FONT;
    ctx.textAlign = 'left';
    ctx.fillText('0 dB', barX + barW + 4, y0 + 8);
    ctx.fillText(`-${DYN_RANGE_DB / 2}`, barX + barW + 4, y0 + h / 2 + 4);
    ctx.fillText(`-${DYN_RANGE_DB}`, barX + barW + 4, y0 + h);
  }

  // ------------------------------------------------------------------ hover

  private onHover(e: MouseEvent): void {
    const d = this.data;
    if (!d) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const { x0, y0, w, h } = this.plotRect();
    if (px < x0 || px > x0 + w || py < y0 || py > y0 + h) return;

    const F = d.num_frames;
    const B = d.num_bins;
    const j = Math.min(F - 1, Math.max(0, Math.floor(((px - x0) / w) * F)));
    const i = Math.min(B - 1, Math.max(0, Math.floor((1 - (py - y0) / h) * B)));
    this.hover = { j, i };

    const mag = d.magnitude[j][i];
    const db = 20 * Math.log10(Math.max(mag, 1e-12) / this.maxMag);
    this.readoutEl.textContent =
      `t = ${d.times[j].toFixed(3)} s　f = ${d.frequencies[i].toFixed(1)} Hz　` +
      `幅度 = ${mag.toFixed(4)}（相对峰值 ${db.toFixed(1)} dB）`;
    this.draw();
  }
}
