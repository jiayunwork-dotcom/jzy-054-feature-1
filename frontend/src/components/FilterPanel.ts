// Frequency-domain filtering: choose lowpass/highpass/bandpass, fill the
// cutoff fields either by hand or by brushing the magnitude plot, ask the
// backend to zero out-of-band bins and inverse transform, then overlay the
// result (red dashed) on the time-domain plot.

import { api } from '../api';
import { store } from '../store';
import type { FilterMode } from '../types';

export class FilterPanel {
  private root: HTMLElement;
  private modeSelect!: HTMLSelectElement;
  private loInput!: HTMLInputElement;
  private hiInput!: HTMLInputElement;
  private applyBtn!: HTMLButtonElement;
  private clearBtn!: HTMLButtonElement;
  private statusEl!: HTMLElement;

  constructor(root: HTMLElement) {
    this.root = root;
    this.render();
    this.wire();
    // Listen to brush events from the magnitude plot.
    store.onBus((e) => {
      if (e.type === 'brush') {
        this.loInput.value = e.lo.toFixed(2);
        this.hiInput.value = e.hi.toFixed(2);
        this.onModeChange();
      } else if (e.type === 'brush-clear') {
        /* keep current values; user can clear explicitly */
      }
    });
  }

  private render(): void {
    this.root.innerHTML = `
      <h2>④ 频域滤波</h2>
      <div class="row wrap gap">
        <label>类型
          <select id="filter-mode">
            <option value="lowpass">低通</option>
            <option value="highpass">高通</option>
            <option value="bandpass" selected>带通</option>
          </select>
        </label>
        <label>下界 Hz <input id="filter-lo" type="number" min="0" step="1" value="10"/></label>
        <label>上界 Hz <input id="filter-hi" type="number" min="0" step="1" value="60"/></label>
        <button class="btn" id="filter-apply" type="button">应用滤波并逆变换</button>
        <button class="btn ghost" id="filter-clear" type="button">移除叠加</button>
      </div>
      <div class="hint">在上方幅度谱上按住鼠标拖选频段，可自动填入上下界；
        后端把频段外的 DFT 分量置零再做 IDFT。</div>
      <div class="status" id="filter-status"></div>`;
    this.modeSelect = this.root.querySelector('#filter-mode')!;
    this.loInput = this.root.querySelector('#filter-lo')!;
    this.hiInput = this.root.querySelector('#filter-hi')!;
    this.applyBtn = this.root.querySelector('#filter-apply')!;
    this.clearBtn = this.root.querySelector('#filter-clear')!;
    this.statusEl = this.root.querySelector('#filter-status')!;
  }

  private wire(): void {
    this.modeSelect.addEventListener('change', () => this.onModeChange());
    this.applyBtn.addEventListener('click', () => void this.apply());
    this.clearBtn.addEventListener('click', () => {
      store.update({ filteredSignal: null });
      this.statusEl.textContent = '已移除滤波叠加。';
    });
    this.onModeChange();
  }

  private onModeChange(): void {
    const mode = this.modeSelect.value as FilterMode;
    this.loInput.disabled = mode === 'lowpass';
    this.hiInput.disabled = mode === 'highpass';
  }

  private async apply(): Promise<void> {
    const { signal, fs } = store.state;
    const mode = this.modeSelect.value as FilterMode;
    const lo = parseFloat(this.loInput.value);
    const hi = parseFloat(this.hiInput.value);
    const cutoffLow = mode === 'lowpass' ? null : Number.isFinite(lo) ? lo : null;
    const cutoffHigh = mode === 'highpass' ? null : Number.isFinite(hi) ? hi : null;

    this.statusEl.textContent = '滤波与逆变换计算中…';
    try {
      const res = await api.filter(signal, fs, mode, cutoffLow, cutoffHigh);
      store.update({ filteredSignal: res.filtered_signal });
      this.statusEl.innerHTML =
        `✅ 完成。输入能量 ${res.input_energy.toFixed(3)} → ` +
        `输出能量 ${res.output_energy.toFixed(3)}；` +
        `带外高频残余能量 <b>${res.residual_high_energy.toExponential(2)}</b>`;
    } catch (err) {
      this.statusEl.textContent = `⚠ ${(err as Error).message}`;
    }
  }
}
