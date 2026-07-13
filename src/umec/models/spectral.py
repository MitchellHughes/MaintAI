"""Spectral features from ranked reduction-statistic differences (UMEC paper step 3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpectralBasis:
    """Per ECOC-bit spectral model fitted on training rank features."""

    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    third_moment: np.ndarray | None = None


def rank_difference_features(reduction_stack: np.ndarray) -> np.ndarray:
    """
    Build rank features from per-classifier reduction statistics for one ECOC bit.

    Parameters
    ----------
    reduction_stack : array (n_samples, n_classifiers)
    """
    n_samples, n_classifiers = reduction_stack.shape
    if n_classifiers == 0:
        return np.zeros((n_samples, 0))

    raw_ranked = np.zeros((n_samples, n_classifiers), dtype=float)
    for col in range(n_classifiers):
        order = np.argsort(np.argsort(reduction_stack[:, col]))
        raw_ranked[:, col] = order / max(n_samples - 1, 1)

    if n_classifiers < 2:
        return raw_ranked

    diffs = []
    for i in range(n_classifiers):
        for j in range(i + 1, n_classifiers):
            diffs.append(reduction_stack[:, i] - reduction_stack[:, j])
    diff_matrix = np.column_stack(diffs)

    ranked_diffs = np.zeros_like(diff_matrix)
    for col in range(diff_matrix.shape[1]):
        order = np.argsort(np.argsort(diff_matrix[:, col]))
        ranked_diffs[:, col] = order / max(n_samples - 1, 1)

    return np.hstack([raw_ranked, ranked_diffs])


def fit_spectral_basis(
    features: np.ndarray,
    n_components: int = 3,
    use_third_moment: bool = True,
) -> SpectralBasis:
    """Fit second-order spectral basis; optionally retain third-moment skew vector."""
    if features.size == 0:
        raise ValueError("Cannot fit spectral basis on empty features.")

    mean = features.mean(axis=0)
    centered = features - mean
    n_samples = centered.shape[0]

    if n_samples < 2:
        cov = np.eye(centered.shape[1])
    else:
        cov = (centered.T @ centered) / max(n_samples - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    k = min(n_components, eigvecs.shape[1])
    components = eigvecs[:, :k]
    eigenvalues = eigvals[:k]

    third = None
    if use_third_moment and n_samples > 2:
        std = centered.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        third = np.mean((centered / std) ** 3, axis=0)

    return SpectralBasis(mean=mean, components=components, eigenvalues=eigenvalues, third_moment=third)


def project_spectral(features: np.ndarray, basis: SpectralBasis) -> np.ndarray:
    centered = features - basis.mean
    projection = centered @ basis.components  # (n_samples, k)

    if basis.third_moment is not None and basis.third_moment.size:
        # Project third-moment-weighted features onto the same k components
        # instead of concatenating the full n_features vector
        third_weighted = centered * basis.third_moment          # (n_samples, n_features)
        third_projected = third_weighted @ basis.components     # (n_samples, k)
        projection = projection + third_projected               # additive correction, stays (n_samples, k)

    return projection  # always (n_samples, k) — matches len(basis.eigenvalues)
