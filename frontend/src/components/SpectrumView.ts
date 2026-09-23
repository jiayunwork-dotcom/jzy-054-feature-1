// Center analysis view: asks the backend for windowed DFT spectra and draws
// three stacked plots — magnitude, phase, power — with the frequency axis in
// real Hz (fs-derived). Dragging on the magnitude plot selects a frequency
// band for the filter module.

import { api } from '../api';
import type { DftResponse, WindowName } from '../types';
import { store } from '../store';
import { Plot } from '../util/plot';
import { WINDOW_HEX } from '../util/colors';

export class SpectrumView {
  private root: HTMLElement;
  private magHost!: HTMLElement;
  private phaseHost!: HTMLElement;
  private powHost!: HTMLElement;
  private statusEl!: HTMLElement;
  private magPlot!: Plot;
  private phasePlot!: Plot;
  private powPlot!: Plot;
  private data: DftResponse | null = null;
  private inFlight = 0;
  private debounceTimer: number | null = null;

  // brush state (Hz)
  private brushLo: number | null = null;
  private brushHi: number | null = null;
  private dragging = false;

  constructor(root: HTMLElement) {
    this.root = root;
    this.renderShell();
    this.magPlot = new Plot(this.magHost, 190, { xLabel: '频率 (Hz)', yLabel: '幅度' });
    this.phasePlot = new Plot(this.phaseHost, 130, { xLabel: '频率 (Hz)', yLabel: '相位 rad' });
    this.powPlot = new Plot(this.powHost, 170, { xLabel: '频率 (Hz)', yLabel: '功率 |X|²' });
    this.attachBrush();
    store.subscribe(() => this.scheduleFetch());
    window.addEventListener('resize', () => this.renderData());
    this.scheduleFetch();
  }

  private renderShell(): void {
    this.root.innerHTML = `
      <div class="plot-stack">
        <div class="plot-label">幅度谱（正半轴 0 … fs/2，不同窗用不同颜色叠加）</div>
        <div id="mag-host"></div>
        <div class="plot-label">相位谱（仅画第一个选中窗；幅度过小处相位无意义故留白）</div>
        <div id="phase-host"></div>
        <div class="plot-label">功率谱</div>
        <div id="pow-host"></div>
      </div>
      <div class="status" id="spectrum-status"></div>`;
    this.magHost = this.root.querySelector('#mag-host')!;
    this.phaseHost = this.root.querySelector('#phase-host')!;
    this.powHost = this.root.querySelector('#pow-host')!;
    this.statusEl = this.root.querySelector('#spectrum-status')!;
  }

  // ------------------------------------------------------------- data fetching

  private scheduleFetch(): void {
    if (this.debounceTimer != null) window.clearTimeout(this.debounceTimer);
    this.debounceTimer = window.setTimeout(() => {
      void this.fetchData();
    }, 80);
  }

  private async fetchData(): Promise<void> {
    const { signal, fs, n, paddedN, windows } = store.state;
    if (signal.length === 0) return;
    const token = ++this.inFlight;
    this.statusEl.textContent = '频谱计算中…';
    try {
      const data = await api.dft(signal, fs, n, paddedN, windows);
      if (token !== this.inFlight) return; // a newer request superseded this one
      this.data = data;
      this.renderData();
      const padNote =
        paddedN && paddedN > signal.length
          ? `；已补零到 N′=${paddedN}（频谱被插值细化，bin 间距 ${(fs / paddedN).toFixed(2)} Hz）`
          : `；bin 间距 ${(fs / data.n_padded).toFixed(2)} Hz`;
      this.statusEl.textContent =
        `N=${data.n_padded}，fs=${fs} Hz，奈奎斯特频率 ${fs / 2} Hz${padNote}`;
    } catch (err) {
      if (token === this.inFlight) this.statusEl.textContent = `⚠ ${(err as Error).message}`;
    }
  }

  // ------------------------------------------------------------------ drawing

  private renderData(): void {
    if (!this.data) return;
    const d = this.data;
    const half = Math.floor(d.n_padded / 2) + 1;
    const fs = d.fs;
    const freqs = d.frequencies.slice(0, half);
    const xRange = { min: 0, max: fs / 2 };

    this.magPlot.resize();
    this.phasePlot.resize();
    this.powPlot.resize();

    let magMax = 1e-12;
    let powMax = 1e-12;
    for (const spec of d.spectra) {
      for (let i = 0; i < half; i++) {
        if (spec.magnitude[i] > magMax) magMax = spec.magnitude[i];
        if (spec.power[i] > powMax) powMax = spec.power[i];
      }
    }
    this.magPlot.beginFrame(xRange, { min: 0, max: magMax * 1.08 });
    this.powPlot.beginFrame(xRange, { min: 0, max: powMax * 1.08 });
    this.phasePlot.beginFrame(xRange, { min: -Math.PI, max: Math.PI });

    // Nyquist marker.
    this.magPlot.vline(fs / 2, 'rgba(255,255,255,0.4)', [4, 4]);
    this.phasePlot.vline(fs / 2, 'rgba(255,255,255,0.4)', [4, 4]);
    this.powPlot.vline(fs / 2, 'rgba(255,255,255,0.4)', [4, 4]);

    d.spectra.forEach((spec) => {
      const color = WINDOW_HEX[spec.window as WindowName] ?? '#ffffff';
      const mag = spec.magnitude.slice(0, half);
      const pow = spec.power.slice(0, half);
      this.magPlot.line(freqs, mag, color, 1.6);
      this.powPlot.line(freqs, pow, color, 1.5);
    });

    // Phase only for the first selected window, masked where magnitude is low.
    const first = d.spectra[0];
    const phaseMasked = first.phase.slice(0, half).map((p, i) =>
      first.magnitude[i] > 0.02 * magMax ? p : NaN,
    );
    this.phasePlot.line(freqs, phaseMasked, WINDOW_HEX[first.window as WindowName], 1.4);

    // Brush overlay.
    if (this.brushLo != null && this.brushHi != null) {
      this.magPlot.hspan(this.brushLo, this.brushHi, 'rgba(255,209,102,0.18)');
      this.magPlot.vline(this.brushLo, '#ffd166');
      this.magPlot.vline(this.brushHi, '#ffd166');
    }
  }

  // ------------------------------------------------------------- brush select

  private attachBrush(): void {
    const c = this.magPlot.canvas;
    c.style.cursor = 'crosshair';
    c.addEventListener('mousedown', (e) => {
      const rect = c.getBoundingClientRect();
      const f = this.magPlot.pxToDataX(e.clientX - rect.left);
      this.dragging = true;
      this.brushLo = Math.max(0, Math.min(this.storeFs() / 2, f));
      this.brushHi = this.brushLo;
      this.renderData();
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!this.dragging) return;
      const rect = this.magPlot.canvas.getBoundingClientRect();
      const f = this.magPlot.pxToDataX(e.clientX - rect.left);
      this.brushHi = Math.max(0, Math.min(this.storeFs() / 2, f));
      this.renderData();
    });
    window.addEventListener('mouseup', () => {
      if (!this.dragging) return;
      this.dragging = false;
      if (this.brushLo != null && this.brushHi != null) {
        let lo = Math.min(this.brushLo, this.brushHi);
        let hi = Math.max(this.brushLo, this.brushHi);
        if (hi - lo < 0.5) {
          // Treat a click (no drag) as "clear selection".
          lo = 0;
          hi = 0;
          this.brushLo = null;
          this.brushHi = null;
          this.renderData();
          store.emit({ type: 'brush-clear' });
          return;
        }
        this.brushLo = lo;
        this.brushHi = hi;
        this.renderData();
        store.emit({ type: 'brush', lo, hi });
      }
    });
  }

  private storeFs(): number {
    return store.state.fs;
  }

  clearBrush(): void {
    this.brushLo = null;
    this.brushHi = null;
    this.renderData();
  }
}
