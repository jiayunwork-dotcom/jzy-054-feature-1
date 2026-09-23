// Window comparison controls: which windows are overlaid in the spectra,
// the Kaiser beta, and the table of main-lobe width / peak sidelobe level.

import { api } from '../api';
import { store } from '../store';
import type { WindowName, WindowSelection } from '../types';

const WINDOW_LABELS: Record<WindowName, string> = {
  rect: '矩形窗',
  hann: '汉宁窗',
  hamming: '汉明窗',
  blackman: '布莱克曼窗',
  kaiser: '凯泽窗',
};

const ALL_NAMES: WindowName[] = ['rect', 'hann', 'hamming', 'blackman', 'kaiser'];

export class WindowCompare {
  private root: HTMLElement;
  private tableBody!: HTMLElement;
  private betaWrap!: HTMLElement;

  constructor(root: HTMLElement) {
    this.root = root;
    this.render();
    this.refreshTable();
  }

  private render(): void {
    this.root.innerHTML = `
      <div class="window-controls">
        <div class="window-checks" id="win-checks"></div>
        <label class="kaiser-beta" id="beta-wrap" hidden>
          凯泽 β
          <input id="kaiser-beta" type="number" min="0" step="0.1" value="5" />
        </label>
      </div>
      <table class="metrics-table">
        <thead>
          <tr><th>窗函数</th><th>主瓣宽度 (bin)</th><th>最高旁瓣 (dB)</th></tr>
        </thead>
        <tbody id="metrics-body"></tbody>
      </table>`;
    this.tableBody = this.root.querySelector('#metrics-body')!;
    this.betaWrap = this.root.querySelector('#beta-wrap')!;

    const checks = this.root.querySelector('#win-checks')!;
    ALL_NAMES.forEach((name) => {
      const label = document.createElement('label');
      label.className = 'window-check';
      label.innerHTML = `<input type="checkbox" data-name="${name}"
        ${name === 'rect' ? 'checked' : ''}/><span>${WINDOW_LABELS[name]}</span>`;
      checks.appendChild(label);
      label.querySelector('input')!.addEventListener('change', () =>
        this.onChecks(),
      );
    });

    this.root.querySelector('#kaiser-beta')!.addEventListener('input', (e) => {
      const beta = parseFloat((e.target as HTMLInputElement).value);
      const windows = store.state.windows.map((w) =>
        w.name === 'kaiser' ? { ...w, beta: Number.isFinite(beta) ? beta : 0 } : w,
      );
      store.update({ windows });
      this.refreshTable();
    });
  }

  private onChecks(): void {
    const selected: WindowSelection[] = [];
    const checks = this.root.querySelectorAll<HTMLInputElement>('input[data-name]');
    let beta = 5;
    const existingKaiser = store.state.windows.find((w) => w.name === 'kaiser');
    if (existingKaiser?.beta != null) beta = existingKaiser.beta;
    checks.forEach((input) => {
      if (input.checked) {
        const name = input.dataset.name as WindowName;
        selected.push({ name, beta: name === 'kaiser' ? beta : null });
      }
    });
    if (selected.length === 0) {
      // Never leave the analysis without a window: undo the last uncheck.
      const previous = store.state.windows[0].name;
      const box = this.root.querySelector<HTMLInputElement>(
        `input[data-name="${previous}"]`,
      );
      if (box) box.checked = true;
      return;
    }
    this.betaWrap.hidden = !selected.some((w) => w.name === 'kaiser');
    store.update({ windows: selected });
  }

  /** Ask the backend for table figures whenever the window set changes. */
  private tableSignature = '';
  async refreshTable(): Promise<void> {
    const selections = store.state.windows;
    const sig = JSON.stringify(selections);
    if (sig === this.tableSignature) return;
    this.tableSignature = sig;
    try {
      const { windows } = await api.windows(128, selections);
      this.tableBody.innerHTML = windows
        .map((w) => {
          const beta = w.beta != null ? ` (β=${w.beta})` : '';
          const ml = w.mainlobe_bins != null ? w.mainlobe_bins.toFixed(2) : '—';
          const sl =
            w.peak_sidelobe_db != null ? w.peak_sidelobe_db.toFixed(1) : '—';
          return `<tr><td>${WINDOW_LABELS[w.name]}${beta}</td><td>${ml}</td><td>${sl}</td></tr>`;
        })
        .join('');
    } catch (err) {
      this.tableBody.innerHTML = `<tr><td colspan="3" class="error-text">${
        (err as Error).message
      }</td></tr>`;
    }
  }
}
