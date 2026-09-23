"""Window generation and validation tests."""

import numpy as np
import pytest

from app.errors import BadRequest
from app.windows import _i0, get_window


@pytest.mark.parametrize("name", ["rect", "hann", "hamming", "blackman"])
def test_window_basic_properties(name):
    w = get_window(name, 128)
    assert w.shape == (128,)
    assert np.all(np.isfinite(w))
    # Periodic windows are non-negative up to floating point noise (the
    # Blackman formula lands on -1.4e-17 at the zero endpoints).
    assert np.all(w >= -1e-12)
    assert np.all(w <= 1.0 + 1e-12)
    if name != "rect":
        assert w[0] < 0.1
    # Symmetric: w[n] == w[N-n] (periodic convention).
    for k in range(1, 8):
        assert w[k] == pytest.approx(w[-k], abs=1e-12)


def test_kaiser_matches_known_values():
    # beta=0 reduces to the rectangular window; peak is 1.
    w0 = get_window("kaiser", 64, 0.0)
    np.testing.assert_allclose(w0, np.ones(64))
    w8 = get_window("kaiser", 64, 8.6)
    # Center sample is the peak; periodic-convention endpoints are both small
    # (the duplicated endpoint is excluded, so they need not be equal).
    assert w8[32] == pytest.approx(1.0)
    assert w8[0] < 0.005 and w8[-1] < 0.005


def test_i0_series():
    assert _i0(0.0) == pytest.approx(1.0)
    assert _i0(1.0) == pytest.approx(1.266065877752008, rel=1e-12)
    assert _i0(5.0) == pytest.approx(27.2398718236044, rel=1e-12)
    x = np.array([0.0, 1.0, 5.0])
    np.testing.assert_allclose(
        _i0(x), [1.0, 1.266065877752008, 27.2398718236044], rtol=1e-12
    )


def test_unknown_window_rejected():
    with pytest.raises(BadRequest, match="unknown window"):
        get_window("hanning", 32)


def test_kaiser_requires_beta():
    with pytest.raises(BadRequest, match="beta"):
        get_window("kaiser", 32)


@pytest.mark.parametrize("bad_beta", [-1.0, float("nan"), float("inf")])
def test_kaiser_bad_beta(bad_beta):
    with pytest.raises(BadRequest):
        get_window("kaiser", 32, bad_beta)
