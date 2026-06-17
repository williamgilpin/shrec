"""Lightweight plotting helpers (optional — requires ``shrec[viz]``).

Two functions cover the bulk of the boilerplate a user writes to *see* a SHREC
result: overlay a reconstructed driver on the truth, and look at a recurrence
graph. Imported lazily (``from shrec.plotting import ...``) so the core library
has no hard matplotlib dependency.
"""
import numpy as np

import matplotlib.pyplot as plt


def _aligned(z_true, z_pred):
    """z-score both signals and sign-flip the reconstruction to match the
    truth. SHREC recovers the driver only up to sign and scale (the Fiedler
    eigenvector's sign is arbitrary), so alignment is required before overlay."""
    z_true = np.asarray(z_true, float).ravel()
    z_pred = np.asarray(z_pred, float).ravel()
    zt = (z_true - z_true.mean()) / (z_true.std() + 1e-12)
    zp = (z_pred - z_pred.mean()) / (z_pred.std() + 1e-12)
    if np.dot(zt, zp) < 0:
        zp = -zp
    return zt, zp


def plot_driver_overlay(z_true, z_pred, ax=None, normalize=True):
    """Overlay a reconstructed driver on the ground-truth driver.

    Args:
        z_true, z_pred (array-like): the true and reconstructed driver signals.
        ax (matplotlib Axes or None): axes to draw on; created if None.
        normalize (bool): z-score and sign-align both signals first (default).
            SHREC recovers the driver up to sign/scale, so this is usually what
            you want; set False to plot the raw values.

    Returns:
        matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    if normalize:
        z_true, z_pred = _aligned(z_true, z_pred)
    else:
        z_true = np.asarray(z_true, float).ravel()
        z_pred = np.asarray(z_pred, float).ravel()
    ax.plot(z_true, label="true driver", lw=2, color="k", alpha=0.7)
    ax.plot(z_pred, label="reconstructed", lw=1.5, color="C3")
    ax.set_xlabel("time")
    ax.set_ylabel("driver (normalized)" if normalize else "driver")
    ax.legend(loc="upper right", frameon=False)
    return ax


def plot_recurrence_matrix(A, ax=None, cmap="magma", **imshow_kw):
    """Display a recurrence / consensus affinity matrix as an image.

    Args:
        A (array-like): (T, T) affinity or adjacency matrix.
        ax (matplotlib Axes or None): axes to draw on; created if None.
        cmap (str): colormap.
        **imshow_kw: forwarded to ``ax.imshow``.

    Returns:
        matplotlib Axes.
    """
    A = np.asarray(A)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(A, cmap=cmap, aspect="equal", **imshow_kw)
    ax.set_xlabel("time $j$")
    ax.set_ylabel("time $i$")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="affinity")
    return ax
