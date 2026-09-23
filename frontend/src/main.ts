// Application entry: wire the global controls (N, fs, zero-padding) and
// instantiate the five independently-built feature modules.

import './styles.css';
import { store } from './store';
import { SignalBuilder } from './components/SignalBuilder';
import { WindowCompare } from './components/WindowCompare';
import { SpectrumView } from './components/SpectrumView';
import { AliasDemo } from './components/AliasDemo';
import { FilterPanel } from './components/FilterPanel';

const PAD_CHOICES = [128, 256, 512, 1024, 2048, 4096];

function wireGlobalControls(): void {
  const nSelect = document.querySelector<HTMLSelectElement>('#select-n')!;
  const fsInput = document.querySelector<HTMLInputElement>('#input-fs')!;
  const padSelect = document.querySelector<HTMLSelectElement>('#select-pad')!;

  function rebuildPadChoices(): void {
    const n = store.state.n;
    padSelect.innerHTML =
      `<option value="">不补零 (${n})</option>` +
      PAD_CHOICES.filter((p) => p >= n)
        .map((p) => `<option value="${p}" ${store.state.paddedN === p ? 'selected' : ''}>${p}</option>`)
        .join('');
  }
  rebuildPadChoices();

  nSelect.addEventListener('change', () => {
    const n = parseInt(nSelect.value, 10);
    const patch: { n: number; paddedN: number | null } = { n, paddedN: null };
    if (store.state.paddedN && store.state.paddedN >= n) patch.paddedN = store.state.paddedN;
    store.update(patch);
    rebuildPadChoices();
  });

  fsInput.addEventListener('input', () => {
    const fs = parseFloat(fsInput.value);
    if (Number.isFinite(fs) && fs > 0) store.update({ fs });
  });

  padSelect.addEventListener('change', () => {
    const v = padSelect.value;
    store.update({ paddedN: v ? parseInt(v, 10) : null });
  });
}

function boot(): void {
  wireGlobalControls();

  const builder = new SignalBuilder(document.querySelector('#signal-builder')!);
  const windowCompare = new WindowCompare(document.querySelector('#window-compare')!);
  const spectrum = new SpectrumView(document.querySelector('#spectrum-view')!);
  new AliasDemo(document.querySelector('#alias-demo')!);
  new FilterPanel(document.querySelector('#filter-panel')!);

  // Window-set changes refresh both the table (WindowCompare) and the
  // spectra (SpectrumView subscribes itself).
  store.subscribe(() => void windowCompare.refreshTable());

  // Expose for debugging in the browser console.
  (window as unknown as { __store: unknown }).__store = store;
  void builder;
  void spectrum;
}

boot();
