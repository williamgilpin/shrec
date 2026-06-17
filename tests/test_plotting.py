"""Smoke + contract tests for the optional `shrec.plotting` helpers (plan 0009).

Gated on matplotlib (the `shrec[viz]` extra) so the core suite runs without it.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless

from shrec.plotting import _aligned, plot_driver_overlay, plot_recurrence_matrix


def test_aligned_flips_sign_and_zscores():
    z = np.sin(np.linspace(0, 4 * np.pi, 200))
    zt, zp = _aligned(z, -3.0 * z + 5.0)  # sign-flipped, scaled, offset
    # After alignment the reconstruction tracks the truth, not its negation.
    assert np.corrcoef(zt, zp)[0, 1] > 0.999
    np.testing.assert_allclose(zt.mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(zt.std(), 1.0, atol=1e-6)


def test_plot_driver_overlay_returns_axes_with_two_lines():
    z = np.sin(np.linspace(0, 4 * np.pi, 100))
    ax = plot_driver_overlay(z, -z)
    assert len(ax.get_lines()) == 2
    assert ax.get_legend() is not None


def test_plot_recurrence_matrix_returns_axes_with_image():
    rng = np.random.default_rng(0)
    A = rng.random((20, 20))
    ax = plot_recurrence_matrix(A)
    assert len(ax.get_images()) == 1
