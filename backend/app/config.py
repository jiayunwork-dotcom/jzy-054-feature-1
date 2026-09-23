"""Central configuration: allowed sizes, window names, table constants.

The educational tool only accepts the DFT sizes explicitly named in the
product spec, and those limits are enforced *at the API layer*
(see :mod:`app.api`). The transform kernel itself works for any length so it
can be unit-tested independently.
"""

from __future__ import annotations

# Selectable transform sizes N.
ALLOWED_N: frozenset[int] = frozenset({64, 128, 256, 512, 1024})

# Accepted zero-padding target lengths (must be >= the signal length).
ALLOWED_PADDED_N: frozenset[int] = frozenset(
    {64, 128, 256, 512, 1024, 2048, 4096}
)

WINDOW_NAMES: tuple[str, ...] = (
    "rect",
    "hann",
    "hamming",
    "blackman",
    "kaiser",
)

# Canonical textbook figures shown in the window-comparison table.
# ``mainlobe_bins`` is the main-lobe width measured between the first nulls,
# expressed in DFT bins (i.e. in multiples of 2*pi/N); ``peak_sidelobe_db``
# is the level of the highest sidelobe relative to the main-lobe peak.
# Kaiser values depend on beta and are measured numerically at request time;
# the four fixed entries here are the standard design-table numbers.
WINDOW_TABLE: dict[str, dict[str, float | None]] = {
    "rect": {"mainlobe_bins": 2.0, "peak_sidelobe_db": -13.3},
    "hann": {"mainlobe_bins": 4.0, "peak_sidelobe_db": -31.5},
    "hamming": {"mainlobe_bins": 4.0, "peak_sidelobe_db": -42.7},
    "blackman": {"mainlobe_bins": 6.0, "peak_sidelobe_db": -58.1},
    "kaiser": {"mainlobe_bins": None, "peak_sidelobe_db": None},
}

# Distinct colors reused by the frontend overlay (kept here so both sides can
# agree on a canonical ordering; the frontend mirrors this list).
WINDOW_COLORS: dict[str, str] = {
    "rect": "#e5484d",
    "hann": "#3e9b4f",
    "hamming": "#3b82f6",
    "blackman": "#b58900",
    "kaiser": "#a855f7",
}
