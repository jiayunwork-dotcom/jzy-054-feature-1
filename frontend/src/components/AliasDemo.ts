// Sampling theorem module: sliders for the true tone frequency and sampling
// rate. The backend returns the dense "ground truth" tone, the discrete
// samples, and the sinc-interpolated reconstruction. When f > fs/2 the
// reconstruction is drawn in loud red and the folded apparent frequency is
// reported — driven by the backend's explicit aliasing decision.

import { api } from '../api';
import type { SamplingResponse } from '../types';
import { Plot, autoRange } from '../util/plot';

export class AliasDemo {
  private root: HTMLElement;
  private fInput!: HTMLInputElement;
  private fVal!: HTMLElement;
  private fsInput!: HTMLInputElement;
  private fsVal!: HTMLElement;
  private canvasHost!: HTMLElement;
  private verdict!: HTMLElement;
  private plot!: Plot;
  private data: SamplingResponse | null = null;
  private debounce: number | null = null;

  constructor(root: HTMLElement) {
    this.root = root;
    this.render();
    this.plot = new Plot(this.canvasHost, 220, { xLabel: '时间 t (s)', yLabel: '幅度' });
    this.bind();
    window.addEventListener('resize', () => this.draw());
    void this.fetchData();
  }

  private render(): void {
    this.root.innerHTML = `
      <h2>③ 采样定理与混叠</h2>
      <div class="row wrap gap">
        <label class="slider-label">信号频率
          <input type="range" id="alias-f" min="1" max="160" step="1" value="30"/>
          <span id="alias-f-val" class="slider-val">30 Hz</span>
        </label>
        <label class="slider-label">采样率
          <input type="range" id="alias-fs" min="20" max="200" step="1" value="100"/>
          <span id="alias-fs-val" class="slider-val">100 Hz</span>
        </label>
      </div>
      <div class="verdict" id="alias-verdict"></div>
      <div id="alias-canvas-host"></div>
      <div class="hint">细线：密集采样近似的原始信号；圆点：实际离散采样；
        加粗线：由采样点 sinc 插值重建的信号。采样率低于 2f 时重建线变红。</div>`;
    this.fInput = this.root.querySelector('#alias-f')!;
    this.fVal = this.root.querySelector('#alias-f-val')!;
    this.fsInput = this.root.querySelector('#alias-fs')!;
    this.fsVal = this.root.querySelector('#alias-fs-val')!;
    this.canvasHost = this.root.querySelector('#alias-canvas-host')!;
    this.verdict = this.root.querySelector('#alias-verdict')!;
  }

  private bind(): void {
    const onChange = () => {
      this.fVal.textContent = `${this.fInput.value} Hz`;
      this.fsVal.textContent = `${this.fsInput.value} Hz`;
      if (this.debounce != null) window.clearTimeout(this.debounce);
      this.debounce = window.setTimeout(() => void this.fetchData(), 60);
    };
    this.fInput.addEventListener('input', onChange);
    this.fsInput.addEventListener('input', onChange);
  }

  private async fetchData(): Promise<void> {
    const f = parseFloat(this.fInput.value);
    const fs = parseFloat(this.fsInput.value);
    this.verdict.textContent = '计算中…';
    try {
      this.data = await api.sampling(f, fs, 28);
      this.draw();
      if (this.data.aliased) {
        this.verdict.className = 'verdict aliased';
        this.verdict.textContent =
          `⚠ 混叠！f = ${f} Hz > fs/2 = ${fs / 2} Hz（奈奎斯特频率）。` +
          `采样点无法与 ${this.data.apparent_freq_hz.toFixed(1)} Hz 的信号区分，` +
          `重建信号（红色）按表观频率振荡。`;
      } else {
        this.verdict.className = 'verdict ok';
        this.verdict.textContent =
          `✓ 未混叠：f = ${f} Hz ≤ fs/2 = ${fs / 2} Hz，` +
          `重建信号与原始信号一致。把 fs 拖到 ${2 * f} Hz 以下试试。`;
      }
    } catch (err) {
      this.verdict.className = 'verdict aliased';
      this.verdict.textContent = `⚠ ${(err as Error).message}`;
    }
  }

  private draw(): void {
    if (!this.data) return;
    const d = this.data;
    this.plot.resize();
    const allY = [...d.original_y, ...d.reconstructed_y];
    this.plot.beginFrame(
      { min: d.original_t[0], max: d.original_t[d.original_t.length - 1] },
      autoRange(allY, 0.1),
    );
    // fs/2 annotation verticals are time-domain; instead draw original faint.
    this.plot.line(d.original_t, d.original_y, 'rgba(180,200,230,0.65)', 1.2, [3, 3]);
    this.plot.points(d.sample_t, d.sample_y, '#ffd166', 3.2);
    const reconColor = d.aliased ? '#ff3b3b' : '#2ee59d';
    this.plot.line(d.reconstructed_t, d.reconstructed_y, reconColor, 2.4);
    this.plot.text(
      d.aliased ? '红色 = 混叠后的错误重建' : '绿色 = 正确重建',
      d.original_t[Math.floor(d.original_t.length * 0.02)],
      0.9 * (autoRange(allY).max),
      reconColor,
    );
  }
}
