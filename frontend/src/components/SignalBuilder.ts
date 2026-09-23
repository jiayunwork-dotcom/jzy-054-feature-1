// Left panel: stack up to 8 waveform components, or hand-draw an arbitrary
// waveform with the mouse. The resulting discrete samples are the input to
// every frequency-domain operation.

import { store } from '../store';
import type { Component, WaveformKind } from '../types';
import { Plot, autoRange } from '../util/plot';
import { MAX_COMPONENTS, resampleStroke, synthesize } from '../util/builder';

const KINDS: Array<{ value: WaveformKind; label: string }> = [
  { value: 'sine', label: '正弦' },
  { value: 'cosine', label: '余弦' },
  { value: 'square', label: '方波' },
  { value: 'triangle', label: '三角波' },
  { value: 'sawtooth', label: '锯齿波' },
];

let nextId = 100;

export class SignalBuilder {
  private root: HTMLElement;
  private listEl!: HTMLElement;
  private addBtn!: HTMLButtonElement;
  private canvasHost!: HTMLElement;
  private plot!: Plot;
  private drawBtn!: HTMLButtonElement;
  private clearDrawBtn!: HTMLButtonElement;
  private strokes: Array<[number, number]> = [];
  private drawing = false;
  private statusEl!: HTMLElement;

  constructor(root: HTMLElement) {
    this.root = root;
    this.renderShell();
    this.renderRows();
    store.subscribe(() => this.onStateChange());
    this.attachCanvasInput();
    window.addEventListener('resize', () => this.draw());
    // renderShell already synthesized the initial signal before subscribing,
    // so no listener fired — paint the first frame explicitly.
    this.draw();
  }

  private renderShell(): void {
    this.root.innerHTML = `
      <div class="component-list" id="comp-list"></div>
      <div class="row gap">
        <button class="btn" id="btn-add" type="button">＋ 添加分量</button>
        <button class="btn ghost" id="btn-draw" type="button">✎ 手绘模式：关</button>
        <button class="btn ghost" id="btn-clear-draw" type="button" disabled>清除手绘</button>
      </div>
      <div class="canvas-wrap" id="time-canvas-host"></div>
      <div class="hint">
        横轴为采样点 n（1/fs 秒/点）。手绘模式下在画布中按住鼠标从左向右拖出波形，
        松开后按当前 N 与 fs 离散成采样点。
      </div>
      <div class="status" id="signal-status"></div>`;

    this.listEl = this.root.querySelector('#comp-list')!;
    this.addBtn = this.root.querySelector('#btn-add')!;
    this.drawBtn = this.root.querySelector('#btn-draw')!;
    this.clearDrawBtn = this.root.querySelector('#btn-clear-draw')!;
    this.canvasHost = this.root.querySelector('#time-canvas-host')!;
    this.statusEl = this.root.querySelector('#signal-status')!;
    this.plot = new Plot(this.canvasHost, 260, { xLabel: '采样点 n', yLabel: '幅度' });

    this.addBtn.addEventListener('click', () => this.addComponent());
    this.drawBtn.addEventListener('click', () => this.toggleDrawMode());
    this.clearDrawBtn.addEventListener('click', () => this.clearStroke());

    // Initial discrete signal for the default component stack.
    this.syncSignal();
  }

  // ------------------------------------------------------------ component rows

  private renderRows(): void {
    const { components, drawMode } = store.state;
    this.listEl.innerHTML = '';
    components.forEach((c, idx) => {
      const row = document.createElement('div');
      row.className = 'component-row';
      row.innerHTML = `
        <label class="enable" title="启用/停用该分量">
          <input type="checkbox" data-k="enable" ${c.enabled ? 'checked' : ''}/>
        </label>
        <select data-k="kind">
          ${KINDS.map(
            (k) =>
              `<option value="${k.value}" ${k.value === c.kind ? 'selected' : ''}>${k.label}</option>`,
          ).join('')}
        </select>
        <label>振幅 <input type="number" step="0.1" data-k="amplitude" value="${c.amplitude}"/></label>
        <label>频率Hz <input type="number" min="0" step="0.5" data-k="frequency" value="${c.frequency}"/></label>
        <label>初相rad <input type="number" step="0.1" data-k="phase" value="${c.phase}"/></label>
        <button class="btn tiny danger" data-k="remove" type="button"
          ${components.length <= 1 ? 'disabled' : ''}>删</button>
        <span class="comp-tag">#${idx + 1}</span>`;
      this.listEl.appendChild(row);

      row.querySelector<HTMLInputElement>('[data-k="enable"]')!.addEventListener(
        'change',
        (e) => this.updateComponent(c.id, { enabled: (e.target as HTMLInputElement).checked }),
      );
      row.querySelector<HTMLSelectElement>('[data-k="kind"]')!.addEventListener(
        'change',
        (e) =>
          this.updateComponent(c.id, { kind: (e.target as HTMLSelectElement).value as WaveformKind }),
      );
      (['amplitude', 'frequency', 'phase'] as const).forEach((key) => {
        row.querySelector<HTMLInputElement>(`[data-k="${key}"]`)!.addEventListener(
          'input',
          (e) => {
            const v = parseFloat((e.target as HTMLInputElement).value);
            if (Number.isFinite(v)) this.updateComponent(c.id, { [key]: v });
          },
        );
      });
      row.querySelector('[data-k="remove"]')!.addEventListener('click', () =>
        this.removeComponent(c.id),
      );
    });

    this.addBtn.disabled = drawMode || components.length >= MAX_COMPONENTS;
  }

  private addComponent(): void {
    const { components } = store.state;
    if (components.length >= MAX_COMPONENTS) return;
    const next: Component = {
      id: nextId++,
      enabled: true,
      kind: 'sine',
      amplitude: 0.5,
      frequency: 4 * (components.length + 1),
      phase: 0,
    };
    store.update({ components: [...components, next], filteredSignal: null });
  }

  private removeComponent(id: number): void {
    const components = store.state.components.filter((c) => c.id !== id);
    store.update({ components, filteredSignal: null });
  }
  private updateComponent(id: number, patch: Partial<Component>): void {
    const components = store.state.components.map((c) =>
      c.id === id ? { ...c, ...patch } : c,
    );
    store.update({ components, filteredSignal: null });
    this.syncSignal();
  }

  // ---------------------------------------------------------------- draw mode

  private toggleDrawMode(): void {
    const drawMode = !store.state.drawMode;
    store.update({ drawMode, filteredSignal: null });
    this.drawBtn.textContent = drawMode ? '✎ 手绘模式：开' : '✎ 手绘模式：关';
    this.drawBtn.classList.toggle('active', drawMode);
    this.canvasHost.classList.toggle('draw-mode', drawMode);
    this.clearDrawBtn.disabled = !drawMode && this.strokes.length === 0;
    this.syncSignal();
  }

  private clearStroke(): void {
    this.strokes = [];
    this.clearDrawBtn.disabled = true;
    const zeros = new Array<number>(store.state.n).fill(0);
    store.update({ signal: zeros, filteredSignal: null });
    this.draw();
  }

  private canvasPoint(e: MouseEvent): [number, number] | null {
    const p = this.plot.toData(e.clientX, e.clientY);
    if (!p) return null;
    // Normalize x to [0, 1] over the sample span; clamp the painted value.
    const span = store.state.n - 1;
    const nx = span === 0 ? 0 : p.x / span;
    return [Math.min(1, Math.max(0, nx)), Math.max(-2, Math.min(2, p.y))];
  }

  private attachCanvasInput(): void {
    const c = this.plot.canvas;
    c.addEventListener('mousedown', (e) => {
      if (!store.state.drawMode) return;
      const p = this.canvasPoint(e);
      if (!p) return;
      this.drawing = true;
      this.strokes = [p];
      this.drawStrokePreview();
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!this.drawing) return;
      const p = this.canvasPoint(e);
      if (p && p[0] >= this.strokes[this.strokes.length - 1][0]) this.strokes.push(p);
      this.drawStrokePreview();
    });
    window.addEventListener('mouseup', () => {
      if (!this.drawing) return;
      this.drawing = false;
      if (this.strokes.length > 1) {
        this.clearDrawBtn.disabled = false;
        const signal = resampleStroke(this.strokes, store.state.n);
        store.update({ signal, filteredSignal: null });
      }
    });
  }
  private drawStrokePreview(): void {
    // Show the raw polyline over an empty frame while dragging.
    this.draw();
    if (this.strokes.length > 1) {
      const xs = this.strokes.map(([x]) => x * (store.state.n - 1));
      const ys = this.strokes.map(([, y]) => y);
      this.plot.line(xs, ys, '#ffd166', 2);
    }
  }

  // -------------------------------------------------------------- recomputation

  /** Recompute the discrete samples from whatever the current source is. */
  private syncSignal(): void {
    const { components, n, fs, drawMode } = store.state;
    const signal =
      drawMode && this.strokes.length > 1
        ? resampleStroke(this.strokes, n)
        : drawMode
          ? new Array<number>(n).fill(0)
          : synthesize(components, n, fs);
    store.update({ signal });
  }

  private lastShape = '';

  private onStateChange(): void {
    const s = store.state;
    const shape = `${s.components.length}|${s.drawMode}|${s.n}|${s.fs}`;
    if (shape !== this.lastShape) {
      this.lastShape = shape;
      this.renderRows();
      this.syncSignal();
    }
    this.draw();
  }

  // ------------------------------------------------------------------- drawing

  private draw(): void {
    this.plot.resize();
    const { signal, n, fs, filteredSignal, drawMode } = store.state;
    const xs = signal.map((_, i) => i);
    // While drawing, keep a fixed vertical scale so cursor position matches
    // the amplitude being painted; otherwise auto-scale to the content.
    this.plot.setFixedY(drawMode ? { min: -2, max: 2 } : null);
    const range = autoRange(signal);
    if (filteredSignal) {
      const fr = autoRange(filteredSignal);
      range.min = Math.min(range.min, fr.min);
      range.max = Math.max(range.max, fr.max);
    }
    range.min = Math.min(range.min, -0.2);
    range.max = Math.max(range.max, 0.2);
    this.plot.beginFrame({ min: 0, max: Math.max(1, n - 1) }, range);
    this.plot.line(xs, signal, '#7cc4ff', 1.8);
    // Mark discrete samples so the "sequence" nature stays visible.
    this.plot.points(xs, signal, 'rgba(124,196,255,0.75)', n > 256 ? 1.2 : 2);
    if (filteredSignal) {
      this.plot.line(
        filteredSignal.map((_, i) => i),
        filteredSignal,
        '#ff6b6b',
        1.8,
        [6, 4],
      );
    }
    this.statusEl.textContent =
      `N = ${n}，fs = ${fs} Hz，采样间隔 T = ${(1000 / fs).toFixed(3)} ms，` +
      `时长 ${(n / fs).toFixed(3)} s` +
      (filteredSignal ? '　红色虚线：滤波后信号' : '');
  }
}
