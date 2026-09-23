// Minimal observable store: a single immutable-updated state object plus a
// pub/sub bus for cross-module UI events (brush selections, redraw requests).

import type { AppState, Component } from './types';

export type BusEvent =
  | { type: 'brush'; lo: number; hi: number }
  | { type: 'brush-clear' }
  | { type: 'request-filter-apply' };

export type Listener = () => void;
export type BusListener = (e: BusEvent) => void;

let nextId = 1;

export function createInitialComponents(): Component[] {
  return [
    { id: nextId++, enabled: true, kind: 'sine', amplitude: 1, frequency: 8, phase: 0 },
  ];
}

export function createInitialState(): AppState {
  return {
    signal: new Array<number>(128).fill(0),
    components: createInitialComponents(),
    drawMode: false,
    n: 128,
    fs: 256,
    paddedN: null,
    windows: [{ name: 'rect', beta: null }],
    filteredSignal: null,
  };
}

class Store {
  state: AppState = createInitialState();
  private listeners = new Set<Listener>();
  private busListeners = new Set<BusListener>();

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  onBus(fn: BusListener): () => void {
    this.busListeners.add(fn);
    return () => this.busListeners.delete(fn);
  }

  emit(event: BusEvent): void {
    for (const fn of this.busListeners) fn(event);
  }

  /** Patch state; listeners get one batched notification per call. */
  update(patch: Partial<AppState>): void {
    // Skip reference-identical patches entirely: components call this from
    // within store listeners, so patching a value that didn't actually
    // change must not re-enter listeners.
    const cur = this.state as unknown as Record<string, unknown>;
    const p = patch as unknown as Record<string, unknown>;
    const changed = Object.keys(p).some((k) => cur[k] !== p[k]);
    if (!changed) return;
    this.state = { ...this.state, ...patch };
    for (const fn of this.listeners) fn();
  }

  patchComponents(components: Component[]): void {
    this.update({ components });
  }
}

export const store = new Store();
