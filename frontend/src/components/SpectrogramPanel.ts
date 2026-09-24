// ⑤ Short-time Fourier analysis panel. The signal is sliced into overlapping
// windowed segments; each segment gets its own DFT on the backend and the
// one-sided magnitudes are laid side by side as a 2-D heatmap: time across,
// frequency up, energy as color. Three controls drive everything:
//   frame length N  (time vs frequency resolution trade-off),
//   hop H           (overlap -> smoothness vs compute),
//   window          (spectral leakage vs main-lobe width).
// A built-in linear chirp generator supplies the textbook diagonal line.

import { api } from '../api';
import { store } from '../store';
import type { StftResponse, WindowName } from '../types';
import { Heatmap, type HeatmapData } from '../util/heatmap';

const FRAME_GEARS = [32, 64, 128, 256, 512];
const CHIRP_FS = 256;
const CHIRP_SECONDS = 8;
const CHIRP_N = CHIRP_FS * CHIRP_SECONDS; // 2048 samples
const CHIRP_F0 = 8; // Hz
const DB_FLOOR = -60;

const WINDOW_LABELS: Record<WindowName, string> = {
  rect: '矩形窗',
  hann: '汉宁窗',
  hamming: '汉明窗',
  blackman: '布莱克曼窗',
  kaiser: '凯泽窗',
};

/** Linear chirp sin(2π(f0 t + ½·slope·t²)) sampled at CHIRP_FS. */
function makeChirp(fEnd: number): number[] {
  const out = new Array<number>(CHIRP_N);
  const slope = (fEnd - CHIRP_F0) / CHIRP_SECONDS;
  for (let i = 0; i < CHIRP_N; i++) {
    const t = i / CHIRP_FS;
    out[i] = Math.sin(2 * Math.PI * (CHIRP_F0 * t + 0.5 * slope * t * t));
  }
  return out;
}

type SignalSource = 'chirp' | 'builder';

export class SpectrogramPanel {
  private root: HTMLElement;
  private frameSlider!: HTMLInputElement;
  private frameVal!: HTMLElement;
  private hopSlider!: HTMLInputElement;
  private hopVal!: HTMLElement;
  private windowSelect!: HTMLSelectElement;
  private betaWrap!: HTMLElement;
  private betaInput!: HTMLInputElement;
  private fendInput!: HTMLInputElement;
  private loadChirpBtn!: HTMLButtonElement;
  private useBuilderBtn!: HTMLButtonElement;
  private roundtripBtn!: HTMLButtonElement;
  private resolutionEl!: HTMLElement;
  private statusEl!: HTMLElement;
  private heatmap!: Heatmap;
  private canvasHost!: HTMLElement;

  private frameLength = 128;
  private hopLength = 32;
  private windowName: WindowName = 'hann';
  private beta = 6;
  private source: SignalSource = 'chirp';
  private chirpEnd = 100;
  private chirpSignal: number[] = makeChirp(this.chirpEnd);

  private data: StftResponse | null = null;
  private debounceTimer: number | null = null;
  private inFlight = 0;

  constructor(root: HTMLElement) {
    this.root = root;
    this.renderShell();
    this.heatmap = new Heatmap(this.canvasHost, 320);
    this.bind();
    window.addEventListener('resize', () => this.heatmap.render());
    // The constructed signal may change N/fs/components while this panel
    // shows it; a chirp view is independent of the global store.
    store.subscribe(() => {
      if (this.source === 'builder') this.scheduleFetch();
    });
    this.scheduleFetch();
  }

  private renderShell(): void {
    this.root.innerHTML = `
      <div class="row wrap gap" style="align-items:flex-end">
        <label class="slider-label" style="max-width:230px">
          段长 N（每小段采样点数）
          <input type="range" id="spec-frame" min="0" max="${FRAME_GEARS.length - 1}"
            step="1" value="${FRAME_GEARS.indexOf(this.frameLength)}"/>
          <span class="slider-val" id="spec-frame-val"></span>
        </label>
        <label class="slider-label" style="max-width:230px">
          挪动步长 H（相邻段前移采样点，≤ N）
          <input type="range" id="spec-hop" min="1" max="${this.frameLength}"
            step="1" value="${this.hopLength}"/>
          <span class="slider-val" id="spec-hop-val"></span>
        </label>
        <label>窗函数
          <select id="spec-window">
            ${(Object.keys(WINDOW_LABELS) as WindowName[])
              .map(
                (w) =>
                  `<option value="${w}" ${w === this.windowName ? 'selected' : ''}>${WINDOW_LABELS[w]}</option>`,
              )
              .join('')}
          </select>
        </label>
        <label id="spec-beta-wrap" hidden>凯泽 β
          <input id="spec-beta" type="number" min="0" step="0.5" value="${this.beta}"/>
        </label>
      </div>

      <div class="row wrap gap">
        <button class="btn" id="spec-load-chirp" type="button">
          📈 载入线性扫频测试信号
        </button>
        <label class="chirp-controls">扫频止频
          <input id="spec-fend" type="number" min="10" max="120" step="1"
            value="${this.chirpEnd}"/> Hz（起频 ${CHIRP_F0} Hz，${CHIRP_SECONDS} s，fs=${CHIRP_FS} Hz）
        </label>
        <button class="btn ghost" id="spec-use-builder" type="button">
          使用左侧构造的信号（fs 跟随全局）
        </button>
        <button class="btn ghost" id="spec-roundtrip" type="button">
          🔁 检验分帧往返重建误差
        </button>
      </div>

      <div class="resolution-box" id="spec-resolution"></div>
      <div class="spec-canvas-wrap" id="spec-canvas-host"></div>
      <div class="hint">
        横轴时间、纵轴频率（0 … fs/2），颜色为该时频点的幅度（dB）。
        把鼠标移到图上读取时刻、频率与幅度。两条黄色虚线标出信号两端（t=0 与
        t=时长），虚线外侧淡出来自为无损重建补的边界帧。<br/>
        拖段长体会 <b>时频分辨率互相拽着</b>：段拉长 → 纵向（频率）变细腻、横向（时间）变粗；
        拖步长体会重叠的作用：H 越小重叠越多，画面越平滑、帧数（计算量）越大。
      </div>
      <div class="status" id="spec-status"></div>`;

    this.frameSlider = this.root.querySelector('#spec-frame')!;
    this.frameVal = this.root.querySelector('#spec-frame-val')!;
    this.hopSlider = this.root.querySelector('#spec-hop')!;
    this.hopVal = this.root.querySelector('#spec-hop-val')!;
    this.windowSelect = this.root.querySelector('#spec-window')!;
    this.betaWrap = this.root.querySelector('#spec-beta-wrap')!;
    this.betaInput = this.root.querySelector('#spec-beta')!;
    this.loadChirpBtn = this.root.querySelector('#spec-load-chirp')!;
    this.useBuilderBtn = this.root.querySelector('#spec-use-builder')!;
    this.fendInput = this.root.querySelector('#spec-fend')!;
    this.roundtripBtn = this.root.querySelector('#spec-roundtrip')!;
    this.resolutionEl = this.root.querySelector('#spec-resolution')!;
    this.statusEl = this.root.querySelector('#spec-status')!;
    this.canvasHost = this.root.querySelector('#spec-canvas-host')!;
  }

  private bind(): void {
    this.frameSlider.addEventListener('input', () => {
      this.frameLength = FRAME_GEARS[parseInt(this.frameSlider.value, 10)];
      // Hop must stay a positive integer no larger than the frame.
      this.hopLength = Math.min(this.hopLength, this.frameLength);
      this.hopSlider.max = String(this.frameLength);
      this.hopSlider.value = String(this.hopLength);
      this.syncLabels();
      this.scheduleFetch();
    });

    this.hopSlider.addEventListener('input', () => {
      this.hopLength = parseInt(this.hopSlider.value, 10);
      this.syncLabels();
      this.scheduleFetch();
    });

    this.windowSelect.addEventListener('change', () => {
      this.windowName = this.windowSelect.value as WindowName;
      this.betaWrap.hidden = this.windowName !== 'kaiser';
      this.scheduleFetch();
    });

    this.betaInput.addEventListener('input', () => {
      const v = parseFloat(this.betaInput.value);
      if (Number.isFinite(v) && v >= 0) {
        this.beta = v;
        this.scheduleFetch();
      }
    });

    this.fendInput.addEventListener('input', () => {
      const v = parseFloat(this.fendInput.value);
      if (Number.isFinite(v) && v > CHIRP_F0 && v < CHIRP_FS / 2) {
        this.chirpEnd = v;
      }
    });
    this.loadChirpBtn.addEventListener('click', () => {
      this.chirpSignal = makeChirp(this.chirpEnd);
      this.source = 'chirp';
      this.refreshSourceButtons();
      this.scheduleFetch();
    });
    this.useBuilderBtn.addEventListener('click', () => {
      this.source = 'builder';
      this.refreshSourceButtons();
      this.scheduleFetch();
    });
    this.roundtripBtn.addEventListener('click', () => void this.checkRoundtrip());

    this.heatmap.canvas.addEventListener('mousemove', (e) => {
      const info = this.heatmap.pick(e.clientX, e.clientY);
      this.heatmap.drawHover(info);
    });
    this.heatmap.canvas.addEventListener('mouseleave', () => {
      this.heatmap.drawHover(null);
    });

    this.syncLabels();
    this.refreshSourceButtons();
  }

  private refreshSourceButtons(): void {
    this.loadChirpBtn.classList.toggle('active', this.source === 'chirp');
    this.useBuilderBtn.classList.toggle('active', this.source === 'builder');
  }

  private syncLabels(): void {
    const fs = this.currentFs();
    this.frameVal.textContent =
      `N = ${this.frameLength} 点（时长 ${(this.frameLength / fs).toFixed(3)} s）`;
    this.hopVal.textContent =
      `H = ${this.hopLength} 点（前移 ${(this.hopLength / fs).toFixed(3)} s，` +
      `重叠 ${Math.round((1 - this.hopLength / this.frameLength) * 100)}%）`;
  }

  private currentSignal(): number[] {
    return this.source === 'chirp' ? this.chirpSignal : store.state.signal;
  }

  private currentFs(): number {
    return this.source === 'chirp' ? CHIRP_FS : store.state.fs;
  }

  private windowSpec(): { name: WindowName; beta: number | null } {
    return { name: this.windowName, beta: this.windowName === 'kaiser' ? this.beta : null };
  }

  private scheduleFetch(): void {
    this.syncLabels();
    if (this.debounceTimer != null) window.clearTimeout(this.debounceTimer);
    this.debounceTimer = window.setTimeout(() => void this.fetchData(), 70);
  }

  private async fetchData(): Promise<void> {
    const signal = this.currentSignal();
    const fs = this.currentFs();
    if (signal.length < this.frameLength) {
      this.data = null;
      this.heatmap.setData(this.emptyData());
      this.resolutionEl.innerHTML = '';
      this.statusEl.className = 'status error-text';
      this.statusEl.textContent =
        `⚠ 当前信号只有 ${signal.length} 个采样点，凑不满一整段（N=${this.frameLength}）。` +
        `请把段长调到 ≤ ${signal.length}，或载入更长的扫频信号。`;
      return;
    }
    const token = ++this.inFlight;
    this.statusEl.className = 'status';
    this.statusEl.textContent = '时频计算中…';
    try {
      const data = await api.stft(
        signal,
        fs,
        this.frameLength,
        this.hopLength,
        this.windowSpec(),
      );
      // Only the latest outstanding response owns the view; if it lost the
      // race (parameters moved again), discard it — the newer fetch repaints.
      if (token !== this.inFlight) return;
      this.data = data;
      this.paint();
      this.renderResolution(fs);
      this.statusEl.textContent =
        `${data.num_frames} 帧 × ${data.frequencies.length} 个频率行；` +
        `时间轴 ${data.times[0].toFixed(2)} s → ${data.times[data.times.length - 1].toFixed(2)} s`;
    } catch (err) {
      if (token === this.inFlight) {
        this.statusEl.className = 'status error-text';
        this.statusEl.textContent = `⚠ ${(err as Error).message}`;
      }
    }
  }

  private emptyData(): HeatmapData {
    return {
      db: new Float32Array(0),
      cols: 0,
      rows: 0,
      times: [],
      frequencies: [],
      tMin: 0,
      tMax: 1,
      axisMin: 0,
      axisMax: 1,
      fMax: this.currentFs() / 2,
      dbMin: DB_FLOOR,
      dbMax: 0,
    };
  }


  private paint(): void {
    const d = this.data;
    if (!d) return;
    const fs = d.fs;
    const cols = d.num_frames;
    const rows = d.frequencies.length;
    const db = new Float32Array(cols * rows);
    let maxMag = 1e-12;
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        if (d.magnitude[c][r] > maxMag) maxMag = d.magnitude[c][r];
      }
    }
    // Display window: top pinned at the peak magnitude, floor DB_FLOOR below.
    const dbTop = 20 * Math.log10(maxMag);
    const dbFloor = dbTop + DB_FLOOR;
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        const v = 20 * Math.log10(Math.max(d.magnitude[c][r], 1e-12));
        db[c * rows + r] = Math.max(dbFloor, v);
      }
    }
    this.heatmap.resize();
    // Pad the axis by half a column on each side so the first/last column
    // centers sit inside the box; for a single column use the hop spacing as
    // the half-width so the axis does not collapse.
    const colSpan =
      cols > 1
        ? (d.times[cols - 1] - d.times[0]) / (cols - 1)
        : this.hopLength / fs;
    const tFirst = d.times[0];
    const tLast = d.times[cols - 1];
    this.heatmap.setData({
      db,
      cols,
      rows,
      times: d.times,
      frequencies: d.frequencies,
      tMin: tFirst,
      tMax: tLast,
      axisMin: tFirst - colSpan / 2,
      axisMax: tLast + colSpan / 2,
      fMax: d.frequencies[rows - 1],
      dbMin: dbFloor,
      dbMax: dbTop,
    });
  }

  private renderResolution(fs: number): void {
    const d = this.data;
    if (!d) return;
    const dt = d.time_resolution_s;
    const df = d.frequency_resolution_hz;
    const overlapPct = Math.round(d.overlap_ratio * 100);
    this.resolutionEl.innerHTML = `
      <div class="res-item">
        <span class="res-label">时间分辨率（一列覆盖多宽）</span>
        <span class="res-value">Δt = N/fs = <b>${(dt * 1000).toFixed(2)}</b> ms</span>
      </div>
      <div class="res-item">
        <span class="res-label">频率分辨率（一行对应多少 Hz）</span>
        <span class="res-value">Δf = fs/N = <b>${df.toFixed(3)}</b> Hz</span>
      </div>
      <div class="res-item">
        <span class="res-label">列间距 / 重叠率</span>
        <span class="res-value">H/fs = <b>${((this.hopLength / fs) * 1000).toFixed(2)}</b> ms
          ，重叠 <b>${overlapPct}%</b>，共 <b>${d.num_frames}</b> 帧</span>
      </div>`;
  }

  private async checkRoundtrip(): Promise<void> {
    const signal = this.currentSignal();
    const fs = this.currentFs();
    if (signal.length < this.frameLength) {
      this.statusEl.className = 'status error-text';
      this.statusEl.textContent = '⚠ 信号凑不满一帧，无法做往返重建。';
      return;
    }
    this.statusEl.className = 'status';
    this.statusEl.textContent = '分帧 → IDFT → 加权重叠拼回，计算中…';
    try {
      const forward = await api.stft(
        signal,
        fs,
        this.frameLength,
        this.hopLength,
        this.windowSpec(),
        true,
      );
      const back = await api.istft(
        forward.spectra_real!,
        forward.spectra_imag!,
        this.frameLength,
        this.hopLength,
        signal.length,
        fs,
        this.windowSpec(),
      );
      let maxErr = 0;
      let rms = 0;
      for (let i = 0; i < signal.length; i++) {
        const e = Math.abs(back.signal[i] - signal[i]);
        if (e > maxErr) maxErr = e;
        rms += e * e;
      }
      rms = Math.sqrt(rms / signal.length);
      const cola =
        this.windowName === 'rect'
          ? this.hopLength === this.frameLength
          : this.hopLength <= this.frameLength / 2;
      const note = cola
        ? '当前窗/步长满足 COLA 条件，应机器精度重建。'
        : '当前重叠不满足 COLA（试试汉宁窗 + 50%/75% 重叠，或矩形窗不重叠），边界可能有残差。';
      this.statusEl.innerHTML =
        `🔁 往返重建：最大逐点误差 <b>${maxErr.toExponential(2)}</b>，` +
        `RMS 误差 <b>${rms.toExponential(2)}</b>。${note}`;
    } catch (err) {
      this.statusEl.className = 'status error-text';
      this.statusEl.textContent = `⚠ ${(err as Error).message}`;
    }
  }
}
