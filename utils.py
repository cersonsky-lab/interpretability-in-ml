import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def parity_plot(y_true, y_pred, title="Parity Plot", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))

    # Scatter
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")

    # Diagonal line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    # Metrics
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Inset box (top-left)
    textstr = f"$R^2 = {r2:.2f}$\nRMSE = {rmse:.2f}"
    ax.text(
        0.05, 0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
    )

    # Labels
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)

    # Equal axes
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Grid
    ax.grid(alpha=0.3)

    return ax