"""Matplotlib plots required as evidence."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402


def plot_residuals(y_true, y_pred, path: Path, title: str) -> None:
    """Residual (predicted - actual) vs predicted, and residual histogram."""
    resid = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    ax[0].scatter(y_pred, resid, s=18, alpha=0.7, color="#2b6cb0")
    ax[0].axhline(0, color="black", lw=1)
    ax[0].set_xlabel("Predicted yield (tons)")
    ax[0].set_ylabel("Residual = predicted - actual (tons)")
    ax[0].set_title("Residuals vs predictions")
    ax[1].hist(resid, bins=15, color="#2f855a", edgecolor="white")
    ax[1].axvline(0, color="black", lw=1)
    ax[1].set_xlabel("Residual (tons)")
    ax[1].set_ylabel("Number of test farms")
    ax[1].set_title("Residual distribution")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(cm, path: Path, title: str) -> None:
    """Annotated confusion matrix."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4), dpi=150)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["Not high risk", "High risk"])
    ax.set_yticks([0, 1], ["Not high risk", "High risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_clusters(Z, labels, centers, path: Path, title: str) -> None:
    """2-D PCA projection of the scaled features coloured by cluster."""
    pca = PCA(n_components=2, random_state=0)
    P = pca.fit_transform(Z)
    C = pca.transform(centers)
    fig, ax = plt.subplots(figsize=(6, 4.6), dpi=150)
    sc = ax.scatter(P[:, 0], P[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
    ax.scatter(C[:, 0], C[:, 1], marker="X", s=140, c="black", label="Cluster centre")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)")
    ax.set_title(title)
    ax.legend(*sc.legend_elements(), title="Cluster", loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
